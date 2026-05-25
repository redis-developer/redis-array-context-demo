from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import redis
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

WEB_PREFIX = "web"
CLI_PREFIX = "cli"


def docs_key(filename: str, prefix: str = WEB_PREFIX) -> str:
    """Redis key for the Array storing the document lines."""
    return f"{prefix}:docs:{_slugify(filename)}"


def idx_key(filename: str, prefix: str = WEB_PREFIX) -> str:
    """Redis key / index name for the vector index."""
    return f"{prefix}:idx:{_slugify(filename)}"


def _slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoConfig:
    redis_url: str
    openai_api_key: str
    openai_model: str
    markdown_file: str


def load_config() -> DemoConfig:
    load_dotenv()
    return DemoConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        markdown_file=_require_env("DEMO_MARKDOWN_FILE"),
    )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_redis_client(redis_url: str) -> redis.Redis:
    client = redis.from_url(
        redis_url,
        decode_responses=True,
        socket_keepalive=True,
        socket_connect_timeout=2,
        health_check_interval=0,    # disabled — we pre-warm manually before each timer
    )
    client.ping()  # establish the connection now, not on first command
    return client


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

VECTOR_DIM = 1536  # text-embedding-3-small / ada-002 dimension


def ingest_document(
    redis_client: redis.Redis,
    embeddings: OpenAIEmbeddings,
    filepath: str,
    prefix: str = WEB_PREFIX,
) -> tuple[str, str]:
    """
    Load a Markdown file into a Redis Array (one element per line) and build
    a vector index over non-blank lines. Returns the Array key and index name.

    If the Array key already exists the function returns immediately without
    re-ingesting, making server startup idempotent.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.read().splitlines()

    array_key = docs_key(filepath, prefix)
    index_name = idx_key(filepath, prefix)

    if redis_client.exists(array_key):
        logger.info("Array key %s already exists — skipping ingestion.", array_key)
        return array_key, index_name

    logger.info("Ingesting %s into Redis Array %s (%d lines)…", filepath, array_key, len(lines))

    # ARINSERT appends one or more values at the Array's internal insert cursor.
    # A fresh key starts with cursor at 0, so sequential ARINSERT calls produce
    # accurate 0-based index positions that match the original file line numbers.
    # Batch in chunks of 500 to stay well within Redis command-size limits.
    BATCH = 500
    for i in range(0, len(lines), BATCH):
        chunk = lines[i:i + BATCH]
        redis_client.execute_command("ARINSERT", array_key, *chunk)

    logger.info("Array ingestion complete. Building vector index %s…", index_name)

    # Build vector index over content-bearing lines only.
    # Exclude blank lines and markdown structural noise (code-fence markers,
    # horizontal rules, bare punctuation) that pollute the semantic index.
    non_blank = [
        (i, line) for i, line in enumerate(lines)
        if line.strip() and not re.match(r'^(`{3,}|~{3,}|-{3,}|={3,})\s*\S*\s*$', line.strip())
    ]
    texts = [line for _, line in non_blank]
    vectors = embeddings.embed_documents(texts)

    schema = IndexSchema.from_dict({
        "index": {"name": index_name, "prefix": f"{index_name}:chunk"},
        "fields": [
            {"name": "line_number", "type": "numeric"},
            {"name": "content", "type": "text"},
            {"name": "embedding", "type": "vector", "attrs": {
                "dims": VECTOR_DIM,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            }},
        ],
    })

    idx = SearchIndex(schema, redis_client=redis_client)
    idx.create(overwrite=False)

    records = [
        {
            "line_number": line_idx,
            "content": line,
            "embedding": np.array(vec, dtype="<f4").tobytes(),
        }
        for (line_idx, line), vec in zip(non_blank, vectors)
    ]
    idx.load(records)

    logger.info("Vector index %s ready (%d chunks).", index_name, len(records))
    return array_key, index_name


def document_ready(
    redis_client: redis.Redis,
    filepath: str,
    prefix: str = WEB_PREFIX,
) -> bool:
    """Return True if both the Array key and vector index exist."""
    array_key = docs_key(filepath, prefix)
    index_name = idx_key(filepath, prefix)
    array_exists = bool(redis_client.exists(array_key))
    try:
        idx = SearchIndex.from_existing(index_name, redis_client=redis_client)
        index_exists = idx is not None
    except Exception:
        index_exists = False
    return array_exists and index_exists


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class ToolTrace:
    tool_name: str
    query: str
    results: list[dict[str, Any]]
    latency_ms: int


@dataclass
class TurnResult:
    user_message: str
    assistant_message: str
    tool_used: str                          # "grep" | "vector" | "fetch" | "both" | "none"
    tool_reasoning: str
    tool_commands: list[str] = field(default_factory=list)  # Redis commands executed
    grep_results: list[dict[str, Any]] = field(default_factory=list)
    grep_latency_ms: float | None = None
    vector_results: list[dict[str, Any]] = field(default_factory=list)
    vector_latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def build_tools(
    redis_client: redis.Redis,
    embeddings: OpenAIEmbeddings,
    array_key: str,
    index_name: str,
) -> list:
    """
    Return the three LangChain tools bound to a specific Redis Array key and
    vector index. Producing tools via a factory keeps the agent stateless at
    the module level while letting the document key vary between CLI and web.
    """

    @tool
    def argrep_search(pattern: str) -> str:
        """
        Use this when the user asks to find lines matching a specific term,
        command, flag, configuration option, error message, heading, or any
        literal string — regardless of how they phrase it.

        Examples of requests that should use this tool:
        - "find all headings"
        - "show lines containing AOF"
        - "find the line with the save directive"
        - "what lines mention replication?"
        - "find every line that starts with a #"

        The pattern supports exact match, glob wildcards (e.g. '## *'), and
        regex patterns. Returns matching lines with their 1-based line numbers.
        """
        try:
            # Pre-warm: get array length (also needed for the range arg).
            # Done outside the timer so connection acquisition cost is excluded.
            array_len = redis_client.execute_command("ARLEN", array_key)
            end_idx = max(int(array_len) - 1, 0)
            match_type, effective_pattern = _effective_argrep_pattern(pattern)
            # Timer starts here — measures only the ARGREP round-trip.
            _t0 = time.perf_counter_ns()
            raw = redis_client.execute_command(
                "ARGREP", array_key, 0, end_idx, match_type, effective_pattern, "WITHVALUES"
            )
            elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
        except Exception as exc:
            return f"Error running ARGREP: {exc}"

        # ARGREP with WITHVALUES returns nested pairs: [[idx, value], [idx, value], …]
        # (Each match is its own 2-element sub-array, not a flat interleaved list.)
        # The array is 0-based internally; add 1 so displayed line numbers are 1-based.
        results = []
        for pair in (raw or []):
            results.append({"line": int(pair[0]) + 1, "content": pair[1]})

        if not results:
            return f"No lines matched pattern '{pattern}'. (latency: {elapsed}ms)"

        lines = "\n".join(f"L{r['line']}: {r['content']}" for r in results)
        return f"Matched {len(results)} line(s) (latency: {elapsed}ms):\n{lines}"

    @tool
    def vector_search(query: str, top_k: int = 5) -> str:
        """
        Use this when the user asks a conceptual question — how something works,
        what something means, the purpose of a feature, or the difference between
        two things.

        Examples of requests that should use this tool:
        - "how does AOF persistence work?"
        - "what is the purpose of the save directive?"
        - "explain the difference between RDB and AOF"
        - "what are the trade-offs of using AOF?"

        Do NOT use for requests that reference specific line numbers, ask to find
        exact terms or patterns, or are structural in nature.
        Returns the most relevant document chunks with similarity scores.
        """
        try:
            # Embedding and index setup happen outside the timer — only the
            # FT.SEARCH round-trip to Redis is measured.
            query_vector = embeddings.embed_query(query)
            vq = VectorQuery(
                vector=query_vector,
                vector_field_name="embedding",
                return_fields=["line_number", "content", "vector_distance"],
                num_results=top_k,
            )
            idx = SearchIndex.from_existing(index_name, redis_client=redis_client)
            redis_client.execute_command("ARLEN", array_key)  # pre-warm before timing
            _t0 = time.perf_counter_ns()
            results_raw = idx.query(vq)
            elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
        except Exception as exc:
            return f"Error running vector search: {exc}"

        if not results_raw:
            return f"No results found for query '{query}'. (latency: {elapsed}ms)"

        lines = []
        for r in results_raw:
            score = round(1 - float(r.get("vector_distance", 1)), 3)
            content = r.get("content", "")
            lines.append(f"{score:.3f} — {content}")

        return f"Top {len(results_raw)} result(s) (latency: {elapsed}ms):\n" + "\n".join(lines)

    @tool
    def fetch_lines(start_line: int, end_line: int) -> str:
        """
        ALWAYS use this when the user mentions a specific line number — even if
        they phrase it casually:
        - "show me line 43"
        - "what is on line 43?"
        - "I want to see line 43 of the document"
        - "show me lines 40 to 50"
        - "line 12 please"

        Also use this to expand context around a line number returned by
        argrep_search — for example, if line 42 matched a heading, fetch lines
        42 to 55 to retrieve the full section body.

        Provide start and end as 1-based line numbers (inclusive), exactly as
        humans count lines in a file. To fetch a single line set start_line and
        end_line to the same value.
        """
        # Convert from 1-based (human) to 0-based (array index)
        start_idx = start_line - 1
        end_idx = end_line - 1
        try:
            # Pre-warm: use execute_command (not ping) so the pool returns the
            # same warm connection for the timed command that follows.
            redis_client.execute_command("ARLEN", array_key)
            if start_idx == end_idx:
                # Timer wraps only the ARGET command round-trip.
                _t0 = time.perf_counter_ns()
                value = redis_client.execute_command("ARGET", array_key, start_idx)
                elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
                raw = [value] if value is not None else []
            else:
                # Timer wraps only the ARGETRANGE command round-trip.
                _t0 = time.perf_counter_ns()
                raw = redis_client.execute_command(
                    "ARGETRANGE", array_key, start_idx, end_idx
                )
                elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
        except Exception as exc:
            return f"Error fetching lines: {exc}"

        if not raw:
            return f"No content found between lines {start_line} and {end_line}. (latency: {elapsed}ms)"

        lines = "\n".join(
            f"L{start_line + i}: {content if content is not None else ''}"
            for i, content in enumerate(raw)
        )
        return f"Lines {start_line}–{end_line} (latency: {elapsed}ms):\n{lines}"

    @tool
    def count_lines() -> str:
        """
        Use this when the user asks how many lines the document has, or wants a
        count or total of lines — e.g. "how many lines are in the doc?",
        "what is the line count?", "count the lines". Uses ARLEN which returns
        the array length in O(1) without scanning any elements.
        """
        try:
            redis_client.execute_command("EXISTS", array_key)  # pre-warm before timing
            _t0 = time.perf_counter_ns()
            count = redis_client.execute_command("ARLEN", array_key)
            elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
        except Exception as exc:
            return f"Error running ARLEN: {exc}"
        return f"Document has {int(count)} lines. (latency: {elapsed}ms)"

    return [argrep_search, vector_search, fetch_lines, count_lines]


# ---------------------------------------------------------------------------
# Agent executor
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a technical assistant helping developers understand Redis documentation.
Answer questions accurately and concisely based on the document you have access to.

Tool selection rules — follow these exactly:

1. If the user asks how many lines the document has, or wants a line count or total
   (e.g. "how many lines?", "count the lines", "what is the line count?"), MUST call
   count_lines. Do NOT use argrep_search or any other tool for this.

2. If the user mentions a specific line number (e.g. "line 12", "show me line 43",
   "what is on line 7"), you MUST call fetch_lines. Do NOT use argrep_search for
   line-number requests.

3. If the user asks to find lines matching a term, pattern, heading, flag, or any
   literal string (e.g. "find lines containing AOF", "show all headings"), use
   argrep_search.

4. If the user asks a conceptual question — how something works, what something
   means, differences between features — use vector_search. Answer directly from
   those results. Do NOT then call fetch_lines as a follow-up.

Never answer positional or structural questions from general knowledge or memory —
always retrieve the actual content with a tool. Do not mention tool names or
implementation details in your answers.
"""


@dataclass
class AgentExecutor:
    llm_with_tools: Any
    tool_map: dict[str, Any]
    array_key: str
    index_name: str


def build_executor(
    config: DemoConfig,
    redis_client: redis.Redis,
    array_key: str,
    index_name: str,
) -> AgentExecutor:
    llm = ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config.openai_api_key,
    )
    tools = build_tools(redis_client, embeddings, array_key, index_name)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    return AgentExecutor(
        llm_with_tools=llm_with_tools,
        tool_map=tool_map,
        array_key=array_key,
        index_name=index_name,
    )


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------

def run_turn(executor: AgentExecutor, user_message: str) -> TurnResult:
    """Run one agent turn and return a structured result with tool traces."""
    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    grep_results: list[dict] = []
    grep_latency_ms: float | None = None
    vector_results: list[dict] = []
    vector_latency_ms: float | None = None
    tool_names_used: list[str] = []
    tool_reasoning = ""
    tool_commands: list[str] = []

    # Agentic loop: invoke → execute tool calls → feed ToolMessages → repeat
    while True:
        response = executor.llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_name: str = tc["name"]
            tool_args: dict = tc["args"]
            tool_id: str = tc["id"]

            tool_names_used.append(tool_name)

            tool_fn = executor.tool_map.get(tool_name)
            observation: str = tool_fn.invoke(tool_args) if tool_fn else f"Error: unknown tool '{tool_name}'"

            latency = _parse_latency(observation)

            if tool_name == "argrep_search":
                pattern = tool_args.get("pattern", "")
                match_type, effective_pattern = _effective_argrep_pattern(pattern)
                call_results = _parse_grep_observation(observation)
                for r in call_results:
                    r["latency_ms"] = latency
                grep_results.extend(call_results)
                grep_latency_ms = (grep_latency_ms or 0) + (latency or 0)
                if not tool_reasoning:
                    tool_reasoning = f"Pattern search for: {pattern}"
                tool_commands.append(
                    f"ARGREP {executor.array_key} 0 … {match_type} {effective_pattern} WITHVALUES"
                )

            elif tool_name == "vector_search":
                query = tool_args.get("query", "")
                top_k = tool_args.get("top_k", 5)
                vector_results = _parse_vector_observation(observation)
                vector_latency_ms = (vector_latency_ms or 0) + (latency or 0)
                if not tool_reasoning:
                    tool_reasoning = f"Semantic search for: {query}"
                tool_commands.append(
                    f'FT.SEARCH {executor.index_name} "*=>[KNN {top_k} @embedding $vec]"'
                )

            elif tool_name == "fetch_lines":
                start = tool_args.get("start_line", 1)
                end = tool_args.get("end_line", 1)
                call_results = _parse_fetch_observation(observation, start)
                for r in call_results:
                    r["latency_ms"] = latency
                grep_results.extend(call_results)
                grep_latency_ms = (grep_latency_ms or 0) + (latency or 0)
                if not tool_reasoning:
                    tool_reasoning = f"Fetching lines {start}–{end} by position."
                s_idx, e_idx = start - 1, end - 1
                if s_idx == e_idx:
                    tool_commands.append(f"ARGET {executor.array_key} {s_idx}")
                else:
                    tool_commands.append(f"ARGETRANGE {executor.array_key} {s_idx} {e_idx}")

            elif tool_name == "count_lines":
                grep_latency_ms = (grep_latency_ms or 0) + (latency or 0)
                if not tool_reasoning:
                    tool_reasoning = "Counting total lines in the document."
                tool_commands.append(f"ARLEN {executor.array_key}")

            messages.append(ToolMessage(content=observation, tool_call_id=tool_id))

    assistant_message = response.content if hasattr(response, "content") else str(response)
    tool_used = _classify_tools(tool_names_used)

    return TurnResult(
        user_message=user_message,
        assistant_message=assistant_message,
        tool_used=tool_used,
        tool_reasoning=tool_reasoning,
        tool_commands=tool_commands,
        grep_results=grep_results,
        grep_latency_ms=grep_latency_ms,
        vector_results=vector_results,
        vector_latency_ms=vector_latency_ms,
    )


# ---------------------------------------------------------------------------
# Observation parsers
# ---------------------------------------------------------------------------

def _parse_latency(observation: str) -> float | None:
    m = re.search(r"latency:\s*([\d.]+)ms", observation)
    return float(m.group(1)) if m else None


def _parse_grep_observation(observation: str) -> list[dict]:
    results = []
    # Use [ \t]* instead of \s* so blank lines (L4: \n) don't absorb the
    # following newline and incorrectly capture the next line's content.
    for m in re.finditer(r"L(\d+):[ \t]*(.*)", observation):
        results.append({"line": int(m.group(1)), "content": m.group(2).strip()})
    return results


def _parse_fetch_observation(observation: str, start_hint: Any) -> list[dict]:
    return _parse_grep_observation(observation)


def _parse_vector_observation(observation: str) -> list[dict]:
    results = []
    for m in re.finditer(r"([\d.]+)\s*—\s*(.*)", observation):
        results.append({"score": float(m.group(1)), "content": m.group(2).strip()})
    return results


def _detect_match_type(pattern: str) -> str:
    """
    Infer the ARGREP match type from the pattern string.

    - RE    if the pattern contains regex-specific characters (^, $, +, (, ), |)
    - GLOB  if the pattern contains glob wildcards (*, ?, [, ])
    - EXACT otherwise
    """
    if re.search(r"[\^$+()|]", pattern):
        return "RE"
    if re.search(r"[*?\[\]]", pattern):
        return "GLOB"
    return "EXACT"


def _effective_argrep_pattern(pattern: str) -> tuple[str, str]:
    """
    Return (match_type, effective_pattern) for an ARGREP call.

    Plain text patterns (no regex or glob chars) are automatically wrapped in
    glob wildcards so they act as a case-insensitive 'contains' search rather
    than requiring the entire line to equal the pattern exactly.

    Examples:
      "AOF"      → ("GLOB", "*AOF*")
      "## *"     → ("GLOB", "## *")        # already a glob
      "^save "   → ("RE",   "^save ")      # regex passthrough
    """
    match_type = _detect_match_type(pattern)
    if match_type == "EXACT":
        return "GLOB", f"*{pattern}*"
    return match_type, pattern


def _classify_tools(names: list[str]) -> str:
    has_grep  = "argrep_search" in names
    has_fetch = "fetch_lines"   in names
    has_vec   = "vector_search" in names
    has_arlen = "count_lines"   in names
    if (has_grep or has_fetch) and has_vec:
        return "both"
    if has_grep:
        return "grep"
    if has_fetch:
        return "fetch"
    if has_vec:
        return "vector"
    if has_arlen:
        return "arlen"
    return "none"
