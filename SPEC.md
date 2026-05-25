# Redis Array Context Demo — Specification

## Overview

This demo showcases the **Array** data type introduced in Redis 8.8, using Markdown grep
as the hero use case. It demonstrates how Redis Arrays give agents a native way to store
and retrieve text where line position and exactness matter — complementing, rather than
replacing, semantic vector search.

### The Core Narrative

Vector search is great when you need semantic similarity. But sometimes an agent needs
the exact line, exact option, exact error message, exact heading, or exact command. Redis
Arrays give Redis a native way to store text where line position matters.

The demo makes this contrast tangible by letting an agent decide at runtime whether to
use exact Array grep (ARGREP) or vector similarity search to answer a question — and
showing the user which path was taken, with full retrieval details.

---

## Surfaces

The demo has two independent but complementary surfaces that connect to the same Redis
instance.

### 1. Web App

A chat interface where a user asks questions about a pre-loaded Markdown document. The
agent answers using either exact Array grep or vector search, and the right panel shows
full observability into which path was taken and what was retrieved.

### 2. CLI

A developer-facing command-line tool for loading Markdown files into Redis Arrays and
querying them directly. The CLI is the hands-on, terminal-native version of the same
underlying operations.

---

## Stack

| Layer | Technology |
|---|---|
| Redis | Redis 8.8-rc1 (via Docker) |
| Backend framework | FastAPI |
| Agent framework | LangChain (`create_tool_calling_agent` + `AgentExecutor`) |
| Vector search | RedisVL |
| Array + ARGREP operations | redis-py |
| LLM | OpenAI (configurable model) |
| Frontend | Vanilla HTML / CSS / JavaScript (Nginx-served) |
| Containerization | Docker Compose |

LangGraph is **not** used. The agent is a standard LangChain tool-calling agent, which
keeps the code simpler and makes `return_intermediate_steps=True` sufficient for full
observability of tool calls.

---

## Project Structure

```
redis-array-context-demo/
├── backend/
│   ├── __init__.py
│   ├── app.py          # FastAPI routes
│   └── agent.py        # LangChain agent, tools, startup loader
├── cli/
│   ├── __init__.py
│   └── main.py         # CLI entry point (Click or Typer)
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── docs/               # Bundled sample Markdown file(s)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   └── test_agent.py
├── data/               # Redis persistence volume (gitignored)
├── images/
├── .env.example
├── .gitignore
├── .python-version
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
├── pyproject.toml
└── SPEC.md
```

---

## Redis Key Scheme

Both surfaces share one Redis instance but use distinct key prefixes to keep the
keyspace readable and avoid cross-contamination.

```
web:docs:{filename}     # Array key — document loaded by the web app at startup
web:idx:{filename}      # Vector index — built from the web-loaded document

cli:docs:{filename}     # Array key — document loaded via the CLI load command
cli:idx:{filename}      # Vector index — built from the CLI-loaded document
```

`{filename}` is the base name of the Markdown file, lowercased and slugified
(e.g., `redis-persistence.md` → `redis-persistence`).

The prefix is **not configurable** — it is implicit per surface. This keeps the
separation automatic and prevents confusion during live demos.

---

## Agent Design

### Framework

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=config.openai_model)
tools = [argrep_search, vector_search, fetch_lines]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
```

`return_intermediate_steps=True` gives back every tool call the agent made — tool name,
input, and output — without any extra instrumentation. This is the data source for the
right panel in the web UI.

### Natural Language Queries

All user input is natural language. The agent is responsible for reasoning about whether
a request is precise and structural (requiring exact retrieval) or conceptual (requiring
semantic search). The agent must never answer positional or structural questions from
general knowledge.

Examples of how natural language maps to tools:

| User says | Agent should use |
|---|---|
| "Show me line 43." | `fetch_lines` |
| "What is on line 43?" | `fetch_lines` |
| "I want to see line 43 of the document." | `fetch_lines` |
| "Find all the headings." | `argrep_search` (glob `## *`) |
| "Show me every line that mentions AOF." | `argrep_search` |
| "Find the exact error message for a missing config key." | `argrep_search` |
| "What lines contain the word 'persistence'?" | `argrep_search` |
| "How does AOF persistence work?" | `vector_search` |
| "What is the purpose of the save directive?" | `vector_search` |
| "Explain the difference between RDB and AOF." | `vector_search` |

The key distinction: if the user is asking *where something is* or *what is at a
specific location*, that is a structural query → use a tool. If the user is asking
*what something means or how it works*, that may be semantic → consider `vector_search`.

### System Prompt

The system prompt covers both persona and the structural-vs-semantic reasoning rule.
Tool selection details live in the tool descriptions.

```
You are a technical assistant helping developers understand Redis documentation.
Answer questions accurately and concisely based on the document you have access to.

When a user asks for a specific line number, asks to find lines matching a pattern,
or requests exact content, you MUST use a tool. Never answer positional or structural
questions from general knowledge or memory — always retrieve the actual content.

When a user asks a conceptual question about how something works or what something
means, use vector_search to find the most relevant sections.

Do not guess. Do not mention tool names or implementation details in your answers.
```

### Tools

#### `argrep_search`

Performs exact pattern matching against the Redis Array using ARGREP. Returns matching
lines with their 0-based index positions.

**Description given to the LLM:**
> Use this when the user asks to find lines matching a specific term, command, flag,
> configuration option, error message, heading, or any literal string — regardless of
> how they phrase it. Examples: "find all headings", "show lines containing AOF",
> "find the line with the save directive", "what lines mention replication?".
> Supports exact match, glob wildcards (e.g. `## *`), and regex patterns.
> Returns matching lines with their line numbers.

**Implementation:** redis-py, issuing ARGREP against the `web:docs:{filename}` key.

**Returns:** list of `{"line": int, "content": str}` objects, plus latency in ms.

---

#### `vector_search`

Performs semantic similarity search against the vector index built from the document.

**Description given to the LLM:**
> Use this when the user asks a conceptual question — how something works, what
> something means, the purpose of a feature, or the difference between two things.
> Examples: "how does AOF work?", "what is the purpose of the save directive?",
> "explain persistence modes". Do NOT use for requests that reference specific line
> numbers, exact terms, or structural patterns.
> Returns the most relevant document chunks with similarity scores.

**Implementation:** RedisVL, querying the `web:idx:{filename}` index.

**Returns:** list of `{"score": float, "content": str}` objects, plus latency in ms.

---

#### `fetch_lines`

Fetches a contiguous range of lines from the Array by index position.

**Description given to the LLM:**
> Use this whenever the user references a specific line number — "show me line 43",
> "what is on line 43?", "I want to see line 43 of the document". Also use this to
> retrieve surrounding context around a line number returned by argrep_search.
> Provide start and end line numbers (0-based, inclusive). To fetch a single line,
> set start and end to the same value.

**Implementation:** redis-py, using the Array range command against the
`web:docs:{filename}` key.

**Returns:** list of `{"line": int, "content": str}` objects.

---

## Web App

### Document Loading

The Markdown document is loaded into a Redis Array at server startup via a FastAPI
lifespan event. One element per line. If the Array key already exists, ingestion is
skipped (idempotent startup). The vector index is created at the same time if it does
not already exist.

The path to the Markdown file is configured via `DEMO_MARKDOWN_FILE` in `.env`.

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/ready` | Readiness check — verifies Redis connection, Array key exists, vector index exists |
| `POST` | `/api/chat` | Run one agent turn |

#### `POST /api/chat`

**Request:**
```json
{
  "message": "What persistence modes does Redis support?"
}
```

**Response:**
```json
{
  "user_message": "What persistence modes does Redis support?",
  "assistant_message": "Redis supports three persistence modes...",
  "tool_used": "grep",
  "tool_reasoning": "The query asks for a specific list of named options.",
  "grep_results": [
    { "line": 42, "content": "## Persistence Modes" },
    { "line": 43, "content": "Redis supports three persistence modes: RDB, AOF, and RDB+AOF." }
  ],
  "grep_latency_ms": 3,
  "vector_results": [],
  "vector_latency_ms": null
}
```

`tool_used` is one of `"grep"` | `"vector"` | `"both"` | `"none"`.
`tool_reasoning` is the agent's explanation extracted from intermediate steps.
Latency fields are `null` when the corresponding tool was not called.

### Frontend Layout

The UI is reused directly from the Redis Agent Memory demo with the following changes:

**`index.html`** — same two-pane layout (`chat-pane` + right panel). The right panel's
three sections are remapped:

| Badge | Heading | Content |
|---|---|---|
| `Tool` | Agent Decision | Which tool was chosen and the agent's reasoning |
| `Grep` | Array Grep Result | Pattern used, matched lines with line numbers, latency |
| `Vec` | Vector Search Result | Query used, top-k chunks with similarity scores, latency |

When a tool was not invoked for a given turn, its panel shows a dimmed "not used this
turn" state.

**`styles.css`** — unchanged. The existing `.memory-list`, `.memory-section`, and
`.section-heading` classes are reused as-is.

**`app.js`** — `renderList()` is reused. Grep results render as
`"L42: ## Persistence Modes"` and vector results as `"0.91 — Redis supports three..."`,
keeping them as plain strings compatible with the existing renderer.

---

## CLI

Entry point: `redis-array-demo` (registered via `pyproject.toml` scripts).

### Commands

#### `load`

Loads a Markdown file into a Redis Array, one element per line. Builds a vector index
from the document. Uses the `cli:` key prefix.

```
redis-array-demo load path/to/file.md [--key-name NAME]
```

Output:
```
Loading redis-persistence.md...
  Lines ingested : 847
  Redis key      : cli:docs:redis-persistence
  Vector index   : cli:idx:redis-persistence
  Duration       : 1.2s
Done.
```

#### `grep`

Runs an ARGREP query against a loaded document. Supports exact match, glob, and regex.

```
redis-array-demo grep "##*" [--key NAME] [--mode glob|exact|regex]
```

Output shows line numbers and matched content, formatted as a table.

#### `search`

Runs a vector similarity search against a loaded document.

```
redis-array-demo search "how does AOF persistence work" [--key NAME] [--top-k 5]
```

Output shows similarity scores and matching chunks.

#### `chat`

Starts an interactive chat session against a loaded document. The agent uses the same
tools as the web app but outputs tool call traces inline in the terminal.

```
redis-array-demo chat [--key NAME]
```

Each response shows:
```
[Tool: grep] Pattern: "##*"  →  3 matches  (2ms)
Answer: Redis supports three persistence modes...
```

---

## Docker Compose

```yaml
services:

  redis-database:
    container_name: redis-database
    hostname: redis-database
    image: redis:8.8-rc1
    volumes:
      - ./data:/data
    environment:
      REDIS_ARGS: --save 30 1
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD-SHELL", "redis-cli ping | grep PONG"]
      interval: 10s
      retries: 5
      start_period: 5s
      timeout: 5s

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    depends_on:
      redis-database:
        condition: service_healthy
    env_file: .env
    ports:
      - "8000:8000"

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    depends_on:
      - backend
    ports:
      - "8080:80"
```

The `depends_on` with `condition: service_healthy` on the backend ensures the startup
loader does not run before Redis is ready to accept commands.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `REDIS_URL` | Yes | Redis connection URL (e.g. `redis://localhost:6379`) |
| `OPENAI_API_KEY` | Yes | API key for OpenAI |
| `OPENAI_MODEL` | No | Model name, defaults to `gpt-4.1-mini` |
| `DEMO_MARKDOWN_FILE` | Yes (web) | Path to the Markdown file pre-loaded at startup |

---

## Ingestion Pipeline

When a Markdown file is loaded (either at web startup or via `cli load`):

1. Read the file and split into lines, preserving blank lines to maintain original
   line numbering.
2. Write to Redis as an Array (`ARSET` or equivalent bulk insert), one element per line.
3. Generate embeddings for non-blank lines (batched).
4. Create a vector index and ingest the embeddings via RedisVL.

Empty lines are stored in the Array (to preserve line number accuracy) but excluded from
the vector index.

---

## Open Items

The following items are deferred pending the web app UI spec:

- Final choice of sample Markdown document(s) to bundle in `docs/`
- Whether the right panel items should be plain strings or structured with styled
  line-number / score badges
- Compare mode (run both tools on every query and show side-by-side) — deferred to
  web app spec
