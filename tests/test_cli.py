from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared mock factories
# ---------------------------------------------------------------------------

def _mock_config():
    cfg = MagicMock()
    cfg.redis_url = "redis://localhost:6379"
    cfg.openai_api_key = "sk-test"
    cfg.openai_model = "gpt-4.1-mini"
    cfg.markdown_file = "docs/redis-persistence.md"
    return cfg


def _mock_redis(arlen=10):
    rc = MagicMock()
    rc.execute_command.return_value = arlen
    rc.exists.return_value = 1
    return rc


def _patch_stack(redis_client=None, config=None):
    """Return a context-manager stack of the patches every CLI command needs."""
    rc = redis_client or _mock_redis()
    cfg = config or _mock_config()
    return [
        patch("cli.main.load_config", return_value=cfg),
        patch("cli.main.get_redis_client", return_value=rc),
    ]


# ---------------------------------------------------------------------------
# load command
# ---------------------------------------------------------------------------

class TestLoadCommand:
    def test_load_succeeds(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("line one\nline two\n")
        rc = _mock_redis(arlen=2)

        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=rc),
            patch("cli.main.ingest_document", return_value=("cli:docs:test", "cli:idx:test")),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["load", str(md)])

        assert result.exit_code == 0
        assert "Done" in result.output

    def test_load_shows_key_and_index(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("hello\n")

        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.ingest_document", return_value=("cli:docs:test", "cli:idx:test")),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["load", str(md)])

        assert "cli:docs:test" in result.output
        assert "cli:idx:test" in result.output

    def test_load_force_deletes_existing_key(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("hello\n")
        rc = _mock_redis()
        rc.exists.return_value = 1

        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=rc),
            patch("cli.main.ingest_document", return_value=("cli:docs:test", "cli:idx:test")),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["load", str(md), "--force"])

        rc.delete.assert_called_once()
        assert result.exit_code == 0

    def test_load_missing_file_exits_nonzero(self):
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["load", "nonexistent.md"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# grep command
# ---------------------------------------------------------------------------

class TestGrepCommand:
    def _run(self, pattern, redis_client=None, file="docs/test.md"):
        rc = redis_client or _mock_redis()
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=rc),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            return runner.invoke(app, ["grep", pattern, "--file", file])

    def test_grep_shows_matches(self):
        rc = _mock_redis(arlen=10)
        rc.execute_command.side_effect = [
            10,                                # ARLEN pre-warm
            [[2, "AOF is fast"], [7, "AOF again"]],  # ARGREP WITHVALUES
        ]
        result = self._run("AOF", redis_client=rc)
        assert result.exit_code == 0
        assert "AOF is fast" in result.output
        assert "AOF again" in result.output

    def test_grep_shows_line_numbers(self):
        rc = _mock_redis()
        rc.execute_command.side_effect = [10, [[4, "some line"]]]
        result = self._run("some", redis_client=rc)
        # 0-based index 4 → 1-based line 5
        assert "5" in result.output

    def test_grep_no_matches_shows_message(self):
        rc = _mock_redis()
        rc.execute_command.side_effect = [10, []]
        result = self._run("NOTFOUND", redis_client=rc)
        assert result.exit_code == 0
        assert "No matches" in result.output

    def test_grep_requires_file_or_key(self):
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["grep", "AOF"])
        assert result.exit_code != 0

    def test_grep_plain_text_wrapped_as_glob(self):
        rc = _mock_redis()
        rc.execute_command.side_effect = [10, []]
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=rc),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            runner.invoke(app, ["grep", "AOF", "--file", "docs/test.md"])

        argrep_call = rc.execute_command.call_args_list[-1]
        assert argrep_call.args[4] == "GLOB"
        assert argrep_call.args[5] == "*AOF*"

    def test_grep_shows_latency(self):
        rc = _mock_redis()
        rc.execute_command.side_effect = [10, [[0, "line"]]]
        result = self._run("line", redis_client=rc)
        # Table title includes latency in µs or ms
        assert "µs" in result.output or "ms" in result.output

    def test_grep_redis_error_exits_nonzero(self):
        rc = _mock_redis()
        rc.execute_command.side_effect = Exception("connection refused")
        result = self._run("AOF", redis_client=rc)
        assert result.exit_code != 0
        assert "ARGREP failed" in result.output


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------

class TestSearchCommand:
    def _run(self, query, results=None, file="docs/test.md"):
        rc = _mock_redis()
        rc.execute_command.return_value = None  # pre-warm EXISTS

        mock_result = results if results is not None else [
            {"line_number": "3", "content": "AOF writes every command", "vector_distance": "0.2"},
        ]

        mock_idx = MagicMock()
        mock_idx.query.return_value = mock_result

        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=rc),
            patch("cli.main.OpenAITextVectorizer") as mock_vec,
            patch("redisvl.index.SearchIndex") as mock_si,
            patch("redisvl.query.VectorQuery"),
        ):
            mock_vec.return_value.embed.return_value = [0.1] * 1536
            mock_si.from_existing.return_value = mock_idx
            return runner.invoke(app, ["search", query, "--file", file])

    def test_search_shows_content(self):
        result = self._run("how does AOF work?")
        assert result.exit_code == 0
        assert "AOF writes every command" in result.output

    def test_search_shows_score(self):
        result = self._run("how does AOF work?")
        # distance 0.2 → score 0.800
        assert "0.800" in result.output

    def test_search_no_results_shows_message(self):
        result = self._run("xyzzy", results=[])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_requires_file_or_key(self):
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["search", "some query"])
        assert result.exit_code != 0

    def test_search_shows_latency(self):
        result = self._run("AOF")
        assert "µs" in result.output or "ms" in result.output


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------

class TestChatCommand:
    def _make_turn_result(self, **kwargs):
        from backend.agent import TurnResult
        defaults = dict(
            user_message="test",
            assistant_message="Here is the answer.",
            tool_used="grep",
            tool_reasoning="Pattern search for: AOF",
            tool_commands=["ARGREP web:docs:test 0 … GLOB *AOF* WITHVALUES"],
            grep_results=[{"line": 5, "content": "AOF content"}],
            grep_latency_ms=0.312,
            vector_results=[],
            vector_latency_ms=None,
        )
        return TurnResult(**{**defaults, **kwargs})

    def test_chat_shows_answer(self):
        turn = self._make_turn_result()
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="find AOF\n")

        assert "Here is the answer." in result.output

    def test_chat_shows_tool_label(self):
        turn = self._make_turn_result(tool_used="grep")
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="find AOF\n")

        assert "Array Grep" in result.output

    def test_chat_shows_redis_command(self):
        turn = self._make_turn_result()
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="find AOF\n")

        assert "ARGREP" in result.output

    def test_chat_shows_latency(self):
        turn = self._make_turn_result(grep_latency_ms=0.312)
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="find AOF\n")

        assert "µs" in result.output or "ms" in result.output

    def test_chat_arlen_tool_label(self):
        turn = self._make_turn_result(tool_used="arlen", grep_latency_ms=0.18,
                                      tool_commands=["ARLEN web:docs:test"])
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="how many lines?\n")

        assert "Array Len" in result.output

    def test_chat_vector_tool_shows_vector_latency(self):
        turn = self._make_turn_result(
            tool_used="vector",
            grep_latency_ms=None,
            vector_latency_ms=1.9,
            grep_results=[],
            tool_commands=['FT.SEARCH web:idx:test "*=>[KNN 5 @embedding $vec]"'],
        )
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="how does AOF work?\n")

        assert "Vector Search" in result.output
        assert "1.9ms" in result.output

    def test_chat_requires_file_or_key(self):
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            result = runner.invoke(app, ["chat"])
        assert result.exit_code != 0

    def test_chat_no_tool_skips_tool_block(self):
        turn = self._make_turn_result(
            tool_used="none",
            grep_results=[],
            grep_latency_ms=None,
            tool_commands=[],
            tool_reasoning="",
        )
        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()),
            patch("cli.main.OpenAITextVectorizer"),
            patch("cli.main.build_executor"),
            patch("cli.main.run_turn", return_value=turn),
        ):
            result = runner.invoke(app, ["chat", "--file", "docs/test.md"], input="hi\n")

        assert "Tool:" not in result.output
        assert "Here is the answer." in result.output


# ---------------------------------------------------------------------------
# Global --redis-url option
# ---------------------------------------------------------------------------

class TestRedisUrlOption:
    def test_custom_redis_url_passed_to_client(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("hello\n")

        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()) as mock_get,
            patch("cli.main.ingest_document", return_value=("cli:docs:test", "cli:idx:test")),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            runner.invoke(app, ["--redis-url", "redis://myhost:6380", "load", str(md)])
            mock_get.assert_called_with("redis://myhost:6380")

    def test_default_redis_url_is_localhost(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("hello\n")

        with (
            patch("cli.main.load_config", return_value=_mock_config()),
            patch("cli.main.get_redis_client", return_value=_mock_redis()) as mock_get,
            patch("cli.main.ingest_document", return_value=("cli:docs:test", "cli:idx:test")),
            patch("cli.main.OpenAITextVectorizer"),
        ):
            runner.invoke(app, ["load", str(md)])
            mock_get.assert_called_with("redis://localhost:6379")
