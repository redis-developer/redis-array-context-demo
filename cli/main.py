from __future__ import annotations

import time
from pathlib import Path

import typer
from redisvl.utils.vectorize import OpenAITextVectorizer
from rich.console import Console
from rich.table import Table

from backend.agent import (
    CLI_PREFIX,
    build_executor,
    docs_key,
    idx_key,
    ingest_document,
    load_config,
    get_redis_client,
    run_turn,
    _effective_argrep_pattern,
)

app = typer.Typer(
    name="redis-array-demo",
    help="CLI for the Redis 8.8 Arrays context demo.",
    add_completion=False,
)
console = Console()

# Shared state set by the global callback
_redis_url: str = "redis://localhost:6379"


@app.callback()
def _global(
    redis_url: str = typer.Option(
        "redis://localhost:6379",
        "--redis-url",
        help="Redis connection URL. Defaults to redis://localhost:6379 for local use.",
        envvar="CLI_REDIS_URL",
    ),
):
    """Redis 8.8 Arrays context demo — CLI tools."""
    global _redis_url
    _redis_url = redis_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_clients():
    config = load_config()
    redis_client = get_redis_client(_redis_url)
    return config, redis_client


def _fmt_latency(ms: float) -> str:
    """Adaptive latency string — mirrors the frontend formatLatency() function."""
    if ms < 1:
        return f"{round(ms * 1000)}µs"
    if ms < 10:
        return f"{ms:.1f}ms"
    return f"{round(ms)}ms"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def load(
    filepath: Path = typer.Argument(..., help="Path to the Markdown file to load."),
    key_name: str | None = typer.Option(None, "--key-name", "-k", help="Override the Redis Array key name."),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if the key already exists."),
):
    """
    Load a Markdown file into a Redis Array (one element per line) and build
    a vector index over its content.

    Uses the 'cli:' key prefix to keep CLI data separate from the web app.
    """
    config, redis_client = _get_clients()
    vectorizer = OpenAITextVectorizer(
        model="text-embedding-3-small",
        api_config={"api_key": config.openai_api_key},
    )
    path_str = str(filepath)
    array_key = key_name if key_name else docs_key(path_str, CLI_PREFIX)

    if force and redis_client.exists(array_key):
        redis_client.delete(array_key)
        console.print(f"[yellow]Deleted existing key {array_key} (--force).[/yellow]")

    # Ingestion involves network + embedding calls — wall-clock time is fine here.
    t0 = time.perf_counter_ns()
    with console.status(f"Loading [bold]{filepath.name}[/bold]…"):
        final_key, final_idx = ingest_document(
            redis_client,
            vectorizer,
            path_str,
            prefix=CLI_PREFIX,
        )
    elapsed_s = (time.perf_counter_ns() - t0) / 1_000_000_000

    line_count = redis_client.execute_command("ARLEN", final_key)

    console.print()
    console.print(f"[bold green]✓ Done[/bold green] in {elapsed_s:.1f}s")
    console.print(f"  Lines ingested : [bold]{line_count}[/bold]")
    console.print(f"  Redis key      : [bold]{final_key}[/bold]")
    console.print(f"  Vector index   : [bold]{final_idx}[/bold]")


@app.command()
def grep(
    pattern: str = typer.Argument(..., help="Pattern to search for (exact, glob, or regex)."),
    filepath: Path = typer.Option(None, "--file", "-f", help="Markdown file whose key to query."),
    key_name: str | None = typer.Option(None, "--key-name", "-k", help="Override the Redis Array key name."),
):
    """
    Run an ARGREP query against a loaded document.

    Supports exact match, glob wildcards (e.g. '## *'), and regex patterns.
    Returns matching lines with their line numbers.
    """
    _, redis_client = _get_clients()

    if not filepath and not key_name:
        console.print("[red]Provide --file or --key-name.[/red]")
        raise typer.Exit(1)

    array_key = key_name if key_name else docs_key(str(filepath), CLI_PREFIX)

    try:
        # Pre-warm: get array length (needed for the range arg, excludes connection cost from timer).
        array_len = redis_client.execute_command("ARLEN", array_key)
        end_idx = max(int(array_len) - 1, 0)
        match_type, effective_pattern = _effective_argrep_pattern(pattern)
        # Timer wraps only the ARGREP round-trip — same as backend.
        _t0 = time.perf_counter_ns()
        raw = redis_client.execute_command(
            "ARGREP", array_key, 0, end_idx, match_type, effective_pattern, "WITHVALUES"
        )
        elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
    except Exception as exc:
        console.print(f"[red]ARGREP failed: {exc}[/red]")
        raise typer.Exit(1)

    # WITHVALUES returns nested pairs: [[idx, value], [idx, value], …]
    # Array is 0-based internally; add 1 for 1-based display.
    results = [(int(pair[0]) + 1, pair[1]) for pair in (raw or [])]

    if not results:
        console.print(f"[yellow]No matches for pattern '{pattern}'.[/yellow]")
        return

    table = Table(
        title=f"ARGREP  {match_type} '{effective_pattern}'  ({_fmt_latency(elapsed)})",
        show_header=True,
    )
    table.add_column("Line", style="bold yellow", justify="right", no_wrap=True)
    table.add_column("Content")
    for line_no, content in results:
        table.add_row(str(line_no), content)

    console.print(table)
    console.print(f"[dim]{len(results)} match(es)[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query."),
    filepath: Path = typer.Option(None, "--file", "-f", help="Markdown file whose index to query."),
    key_name: str | None = typer.Option(None, "--key-name", "-k", help="Override the vector index name."),
    top_k: int = typer.Option(5, "--top-k", "-n", help="Number of results to return."),
):
    """
    Run a vector similarity search against a loaded document.

    Returns the most semantically relevant chunks with similarity scores.
    """
    from redisvl.index import SearchIndex
    from redisvl.query import VectorQuery

    config, redis_client = _get_clients()
    vectorizer = OpenAITextVectorizer(
        model="text-embedding-3-small",
        api_config={"api_key": config.openai_api_key},
    )

    if not filepath and not key_name:
        console.print("[red]Provide --file or --key-name.[/red]")
        raise typer.Exit(1)

    index_name = key_name if key_name else idx_key(str(filepath), CLI_PREFIX)

    try:
        # Embedding and index setup happen outside the timer — same as backend.
        query_vector = vectorizer.embed(query)
        vq = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=["line_number", "content", "vector_distance"],
            num_results=top_k,
        )
        idx = SearchIndex.from_existing(index_name, redis_client=redis_client)
        redis_client.execute_command("EXISTS", index_name)  # pre-warm before timing
        _t0 = time.perf_counter_ns()
        results_raw = idx.query(vq)
        elapsed = round((time.perf_counter_ns() - _t0) / 1_000_000, 3)
    except Exception as exc:
        console.print(f"[red]Vector search failed: {exc}[/red]")
        raise typer.Exit(1)

    if not results_raw:
        console.print(f"[yellow]No results for '{query}'.[/yellow]")
        return

    table = Table(
        title=f"FT.SEARCH  '{query}'  ({_fmt_latency(elapsed)})",
        show_header=True,
    )
    table.add_column("Score", style="bold blue", justify="right", no_wrap=True)
    table.add_column("Line", style="dim", justify="right", no_wrap=True)
    table.add_column("Content")
    for r in results_raw:
        score = round(1 - float(r.get("vector_distance", 1)), 3)
        line_no = r.get("line_number", "?")
        content = r.get("content", "")
        table.add_row(f"{score:.3f}", str(line_no), content)

    console.print(table)


@app.command()
def chat(
    filepath: Path = typer.Option(None, "--file", "-f", help="Markdown file to chat about."),
    key_name: str | None = typer.Option(None, "--key-name", "-k", help="Override the Redis Array key name."),
):
    """
    Start an interactive chat session against a loaded document.

    Uses the same agent, tools, and timing logic as the web app.
    Each response shows which tool was invoked and the Redis round-trip latency.
    """
    config, redis_client = _get_clients()

    if not filepath and not key_name:
        console.print("[red]Provide --file or --key-name.[/red]")
        raise typer.Exit(1)

    array_key = key_name if key_name else docs_key(str(filepath), CLI_PREFIX)
    index_name = key_name if key_name else idx_key(str(filepath), CLI_PREFIX)

    executor = build_executor(config, redis_client, array_key, index_name)

    console.print(f"\n[bold]Redis Array Context Chat[/bold]  [dim]({array_key})[/dim]")
    console.print("[dim]Type your question and press Enter. Ctrl+C to exit.\n[/dim]")

    while True:
        try:
            user_input = typer.prompt("You")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input.strip():
            continue

        with console.status("Thinking…"):
            try:
                result = run_turn(executor, user_input.strip())
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                continue

        # Tool label — matches all tool types the backend can return.
        tool_label = {
            "grep":       "Array Grep",
            "fetch":      "Array Fetch",
            "grep_fetch": "Array Grep + Array Range",
            "arlen":      "Array Len",
            "vector":     "Vector Search",
            "both":       "Grep + Vector",
            "none":       "No tool",
        }.get(result.tool_used, result.tool_used)

        if result.tool_used != "none":
            parts = [f"[bold cyan]Tool:[/bold cyan] {tool_label}"]
            if result.total_latency_ms is not None:
                parts.append(f"array {_fmt_latency(result.total_latency_ms)}")
            if result.vector_latency_ms is not None:
                parts.append(f"vector {_fmt_latency(result.vector_latency_ms)}")
            console.print("  " + "  ·  ".join(parts))
            if result.tool_reasoning:
                console.print(f"  [dim]{result.tool_reasoning}[/dim]")
            for cmd in result.tool_commands:
                console.print(f"  [dim]$[/dim] [bold]{cmd}[/bold]")

        console.print(f"\n[bold]Agent:[/bold] {result.assistant_message}\n")


if __name__ == "__main__":
    app()
