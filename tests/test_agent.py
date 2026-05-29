from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from redis.commands.core import ArrayPredicateType

from backend.agent import (
    CLI_PREFIX,
    WEB_PREFIX,
    _classify_tools,
    _detect_match_type,
    _effective_argrep_pattern,
    _parse_grep_observation,
    _parse_latency,
    _parse_vector_observation,
    _slugify,
    docs_key,
    idx_key,
)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

class TestSluggify:
    def test_strips_extension(self):
        assert _slugify("redis-arrays.md") == "redis-arrays"

    def test_strips_path(self):
        assert _slugify("/some/path/my file.md") == "my-file"

    def test_lowercases(self):
        assert _slugify("MyDoc.md") == "mydoc"

    def test_collapses_special_chars(self):
        assert _slugify("Redis  Arrays!.md") == "redis-arrays"

    def test_strips_leading_trailing_dashes(self):
        assert _slugify("--foo--.md") == "foo"


class TestDocsKey:
    def test_web_prefix(self):
        assert docs_key("redis-arrays.md") == "web:docs:redis-arrays"

    def test_cli_prefix(self):
        assert docs_key("redis-arrays.md", CLI_PREFIX) == "cli:docs:redis-arrays"

    def test_strips_path(self):
        assert docs_key("/data/my-doc.md") == "web:docs:my-doc"

    def test_uppercase_name(self):
        assert docs_key("Redis_Persistence.md") == "web:docs:redis-persistence"


class TestIdxKey:
    def test_web_prefix(self):
        assert idx_key("redis-arrays.md") == "web:idx:redis-arrays"

    def test_cli_prefix(self):
        assert idx_key("redis-arrays.md", CLI_PREFIX) == "cli:idx:redis-arrays"

    def test_docs_and_idx_share_slug(self):
        assert docs_key("foo.md").split(":")[-1] == idx_key("foo.md").split(":")[-1]


# ---------------------------------------------------------------------------
# Pattern detection and effective pattern
# ---------------------------------------------------------------------------

class TestDetectMatchType:
    def test_plain_text_is_match(self):
        assert _detect_match_type("AOF") == ArrayPredicateType.MATCH

    def test_glob_star(self):
        assert _detect_match_type("## *") == ArrayPredicateType.GLOB

    def test_glob_question_mark(self):
        assert _detect_match_type("save ?") == ArrayPredicateType.GLOB

    def test_glob_brackets(self):
        assert _detect_match_type("[rR]eplication") == ArrayPredicateType.GLOB

    def test_regex_caret(self):
        assert _detect_match_type("^save ") == ArrayPredicateType.RE

    def test_regex_dollar(self):
        assert _detect_match_type("yes$") == ArrayPredicateType.RE

    def test_regex_pipe(self):
        assert _detect_match_type("RDB|AOF") == ArrayPredicateType.RE

    def test_regex_parens(self):
        assert _detect_match_type("(rdb|aof)") == ArrayPredicateType.RE

    def test_regex_plus(self):
        assert _detect_match_type("save +") == ArrayPredicateType.RE


class TestEffectiveArGrepPattern:
    def test_plain_text_uses_match_type(self):
        pred_type, pattern = _effective_argrep_pattern("AOF")
        assert pred_type == ArrayPredicateType.MATCH
        assert pattern == "AOF"

    def test_glob_pattern_passed_through(self):
        pred_type, pattern = _effective_argrep_pattern("## *")
        assert pred_type == ArrayPredicateType.GLOB
        assert pattern == "## *"

    def test_regex_passed_through(self):
        pred_type, pattern = _effective_argrep_pattern("^save ")
        assert pred_type == ArrayPredicateType.RE
        assert pattern == "^save "

    def test_plain_with_spaces_uses_match_type(self):
        pred_type, pattern = _effective_argrep_pattern("append only")
        assert pred_type == ArrayPredicateType.MATCH
        assert pattern == "append only"


# ---------------------------------------------------------------------------
# Observation parsers
# ---------------------------------------------------------------------------

class TestParseLatency:
    def test_extracts_integer_ms(self):
        assert _parse_latency("Matched 3 line(s) (latency: 12ms):") == 12.0

    def test_extracts_float_ms(self):
        assert _parse_latency("Lines 1–5 (latency: 0.246ms):") == pytest.approx(0.246)

    def test_returns_none_when_absent(self):
        assert _parse_latency("No latency info here.") is None

    def test_returns_none_on_empty_string(self):
        assert _parse_latency("") is None

    def test_extracts_from_count_lines_observation(self):
        assert _parse_latency("Document has 42 lines. (latency: 0.183ms)") == pytest.approx(0.183)


class TestParseGrepObservation:
    def test_single_line(self):
        obs = "Matched 1 line(s) (latency: 5ms):\nL10: ## Heading"
        results = _parse_grep_observation(obs)
        assert results == [{"line": 10, "content": "## Heading"}]

    def test_multiple_lines(self):
        obs = "Matched 2 line(s) (latency: 5ms):\nL10: ## Heading\nL20: ## Another"
        results = _parse_grep_observation(obs)
        assert len(results) == 2
        assert results[0] == {"line": 10, "content": "## Heading"}
        assert results[1] == {"line": 20, "content": "## Another"}

    def test_empty_observation(self):
        assert _parse_grep_observation("No matches found.") == []

    def test_blank_line_content(self):
        obs = "Lines 3–5 (latency: 1ms):\nL3: hello\nL4: \nL5: world"
        results = _parse_grep_observation(obs)
        assert results[1] == {"line": 4, "content": ""}

    def test_strips_trailing_whitespace_from_content(self):
        obs = "Matched 1 line(s) (latency: 1ms):\nL7: some content   "
        results = _parse_grep_observation(obs)
        assert results[0]["content"] == "some content"


class TestParseVectorObservation:
    def test_extracts_scores_and_content(self):
        obs = "Top 2 result(s) (latency: 8ms):\n0.923 — Redis supports AOF\n0.871 — AOF writes every command"
        results = _parse_vector_observation(obs)
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(0.923)
        assert results[0]["content"] == "Redis supports AOF"
        assert results[1]["score"] == pytest.approx(0.871)

    def test_empty_observation(self):
        assert _parse_vector_observation("No results found.") == []

    def test_strips_content_whitespace(self):
        obs = "Top 1 result(s):\n0.750 —   trimmed content   "
        results = _parse_vector_observation(obs)
        assert results[0]["content"] == "trimmed content"


# ---------------------------------------------------------------------------
# Tool classifier
# ---------------------------------------------------------------------------

class TestClassifyTools:
    def test_grep_only(self):
        assert _classify_tools(["argrep_search"]) == "grep"

    def test_fetch_only(self):
        assert _classify_tools(["fetch_lines"]) == "fetch"

    def test_vector_only(self):
        assert _classify_tools(["vector_search"]) == "vector"

    def test_count_lines(self):
        assert _classify_tools(["count_lines"]) == "arlen"

    def test_grep_and_vector_is_both(self):
        assert _classify_tools(["argrep_search", "vector_search"]) == "both"

    def test_fetch_and_vector_is_both(self):
        assert _classify_tools(["fetch_lines", "vector_search"]) == "both"

    def test_empty_is_none(self):
        assert _classify_tools([]) == "none"

    def test_unknown_tool_is_none(self):
        assert _classify_tools(["unknown_tool"]) == "none"

    def test_grep_and_fetch_together_is_grep_fetch(self):
        assert _classify_tools(["argrep_search", "fetch_lines"]) == "grep_fetch"

    def test_count_and_fetch_together_is_fetch(self):
        # count_lines is used as a lookup step (e.g. "show me the last line") —
        # the result should reflect the real action, which is fetch.
        assert _classify_tools(["count_lines", "fetch_lines"]) == "fetch"


# ---------------------------------------------------------------------------
# build_tools: tool functions (mocked Redis)
# ---------------------------------------------------------------------------

def _make_redis(execute_side_effect=None):
    """Return a MagicMock Redis client with sane defaults."""
    client = MagicMock()
    if execute_side_effect:
        client.execute_command.side_effect = execute_side_effect
    return client


def _make_vectorizer(vector=None):
    v = MagicMock()
    v.embed.return_value = vector or [0.1] * 1536
    v.embed_many.return_value = [vector or [0.1] * 1536]
    return v


def _build_tools(redis_client, array_key="web:docs:test"):
    """Call build_tools with SearchIndex construction patched out."""
    from backend.agent import build_tools
    with patch("backend.agent.SearchIndex") as mock_si:
        mock_si.return_value = MagicMock()
        return build_tools(redis_client, _make_vectorizer(), array_key, "web:idx:test")


class TestFetchLinesTool:
    """Tests for the fetch_lines tool function via build_tools."""

    def _get_tool(self, redis_client, array_key="web:docs:test"):
        return _build_tools(redis_client, array_key)["fetch_lines"]

    def test_single_line_calls_arget(self):
        rc = _make_redis()
        rc.arlen.return_value = 10   # pre-warm
        rc.arget.return_value = "line content"
        tool = self._get_tool(rc)
        result = tool(start_line=5, end_line=5)
        rc.arget.assert_called_once_with("web:docs:test", 4)
        assert "L5: line content" in result

    def test_range_calls_argetrange(self):
        rc = _make_redis()
        rc.arlen.return_value = 10   # pre-warm
        rc.argetrange.return_value = ["line3", "line4", "line5"]
        tool = self._get_tool(rc)
        result = tool(start_line=3, end_line=5)
        rc.argetrange.assert_called_once_with("web:docs:test", 2, 4)
        assert "L3: line3" in result
        assert "L4: line4" in result
        assert "L5: line5" in result

    def test_blank_lines_rendered_empty(self):
        rc = _make_redis()
        rc.arlen.return_value = 10
        rc.argetrange.return_value = ["content", None, "more content"]
        tool = self._get_tool(rc)
        result = tool(start_line=1, end_line=3)
        assert "L2: \n" in result or result.count("L2:") == 1

    def test_includes_latency_in_observation(self):
        rc = _make_redis()
        rc.arlen.return_value = 10
        rc.arget.return_value = "hello"
        tool = self._get_tool(rc)
        result = tool(start_line=1, end_line=1)
        assert "latency:" in result
        assert "ms" in result

    def test_redis_error_returns_error_string(self):
        rc = _make_redis()
        rc.arlen.side_effect = Exception("connection refused")
        tool = self._get_tool(rc)
        result = tool(start_line=1, end_line=1)
        assert "Error" in result

    def test_one_based_to_zero_based_conversion(self):
        """Line 1 should map to index 0, line 10 to index 9."""
        rc = _make_redis()
        rc.arlen.return_value = 10
        rc.arget.return_value = "first line"
        tool = self._get_tool(rc)
        tool(start_line=1, end_line=1)
        rc.arget.assert_called_once_with("web:docs:test", 0)


class TestArgrepSearchTool:
    def _get_tool(self, redis_client, array_key="web:docs:test"):
        return _build_tools(redis_client, array_key)["argrep_search"]

    def test_plain_text_uses_match_predicate(self):
        rc = _make_redis()
        rc.arlen.return_value = 5
        rc.argrep.return_value = []
        tool = self._get_tool(rc)
        tool(pattern="AOF")
        predicates = rc.argrep.call_args[0][3]
        assert predicates[0][0] == ArrayPredicateType.MATCH
        assert predicates[0][1] == "AOF"

    def test_glob_pattern_uses_glob_predicate(self):
        rc = _make_redis()
        rc.arlen.return_value = 5
        rc.argrep.return_value = []
        tool = self._get_tool(rc)
        tool(pattern="## *")
        predicates = rc.argrep.call_args[0][3]
        assert predicates[0][0] == ArrayPredicateType.GLOB
        assert predicates[0][1] == "## *"

    def test_regex_pattern_uses_re_predicate(self):
        rc = _make_redis()
        rc.arlen.return_value = 5
        rc.argrep.return_value = []
        tool = self._get_tool(rc)
        tool(pattern="^save ")
        predicates = rc.argrep.call_args[0][3]
        assert predicates[0][0] == ArrayPredicateType.RE

    def test_results_parsed_correctly(self):
        rc = _make_redis()
        rc.arlen.return_value = 5
        rc.argrep.return_value = [[2, "AOF is fast"], [7, "AOF flushes"]]
        tool = self._get_tool(rc)
        result = tool(pattern="AOF")
        assert "L3: AOF is fast" in result   # 0-based index 2 → line 3
        assert "L8: AOF flushes" in result

    def test_no_matches_returns_message(self):
        rc = _make_redis()
        rc.arlen.return_value = 5
        rc.argrep.return_value = []
        tool = self._get_tool(rc)
        result = tool(pattern="NOTFOUND")
        assert "No lines matched" in result

    def test_includes_latency(self):
        rc = _make_redis()
        rc.arlen.return_value = 5
        rc.argrep.return_value = [[0, "some line"]]
        tool = self._get_tool(rc)
        result = tool(pattern="some")
        assert "latency:" in result


class TestCountLinesTool:
    def _get_tool(self, redis_client, array_key="web:docs:test"):
        return _build_tools(redis_client, array_key)["count_lines"]

    def test_calls_arlen(self):
        rc = _make_redis()
        rc.exists.return_value = 1   # pre-warm
        rc.arlen.return_value = 42
        tool = self._get_tool(rc)
        result = tool()
        rc.arlen.assert_called_once_with("web:docs:test")
        assert "42" in result

    def test_includes_latency(self):
        rc = _make_redis()
        rc.exists.return_value = 1
        rc.arlen.return_value = 10
        tool = self._get_tool(rc)
        result = tool()
        assert "latency:" in result

    def test_redis_error_returns_error_string(self):
        rc = _make_redis()
        rc.exists.side_effect = Exception("ARLEN failed")
        tool = self._get_tool(rc)
        result = tool()
        assert "Error" in result
