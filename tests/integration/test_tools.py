"""
Integration tests for the four agent tool functions against a real Redis
8.8-rc1 container.

vector_search is excluded here — it requires live OpenAI embeddings and is
covered separately by the unit tests that mock the vectorizer.  The focus is
on the Array commands (ARGET, ARGETRANGE, ARGREP, ARLEN) and verifying that
the Python layer translates indices, parses observations, and surfaces errors
correctly against real Redis behaviour.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agent import build_tools

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_ARRAY_KEY = "integration:docs:test"
_INDEX_NAME = "integration:idx:test"

# Six-line fixture document — line numbers are 1-based in the public API.
#
#  L1  # Redis Persistence
#  L2  (blank)
#  L3  Redis supports RDB and AOF.
#  L4  RDB saves snapshots to disk.
#  L5  AOF logs every write command.
#  L6  You can combine both strategies.
_LINES = [
    "# Redis Persistence",
    "",
    "Redis supports RDB and AOF.",
    "RDB saves snapshots to disk.",
    "AOF logs every write command.",
    "You can combine both strategies.",
]


def _make_vectorizer() -> MagicMock:
    v = MagicMock()
    v.embed.return_value = [0.1] * 1536
    v.embed_many.side_effect = lambda texts, **_: [[0.1] * 1536 for _ in texts]
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def populated_array(redis_client):
    """Load _LINES into the Redis Array before each test."""
    redis_client.execute_command("ARINSERT", _ARRAY_KEY, *_LINES)
    # redis_client fixture handles flushall teardown


@pytest.fixture()
def tools(redis_client, populated_array):
    return build_tools(redis_client, _make_vectorizer(), _ARRAY_KEY, _INDEX_NAME)


# ---------------------------------------------------------------------------
# fetch_lines (ARGET / ARGETRANGE)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFetchLinesTool:
    def test_single_line_content(self, tools):
        result = tools["fetch_lines"](start_line=1, end_line=1)
        assert "L1: # Redis Persistence" in result

    def test_range_returns_all_lines(self, tools):
        result = tools["fetch_lines"](start_line=3, end_line=5)
        assert "L3: Redis supports RDB and AOF." in result
        assert "L4: RDB saves snapshots to disk." in result
        assert "L5: AOF logs every write command." in result

    def test_blank_line_included_in_range(self, tools):
        """Blank lines must be present so surrounding line numbers stay correct."""
        result = tools["fetch_lines"](start_line=1, end_line=3)
        assert "L2:" in result

    def test_single_line_uses_arget_not_argetrange(self, redis_client, populated_array):
        """start == end must call ARGET (O(1)), not ARGETRANGE."""
        calls: list[tuple] = []
        original = redis_client.execute_command

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        redis_client.execute_command = spy
        tools = build_tools(redis_client, _make_vectorizer(), _ARRAY_KEY, _INDEX_NAME)
        tools["fetch_lines"](start_line=3, end_line=3)

        array_cmds = [c[0] for c in calls]
        assert "ARGET" in array_cmds
        assert "ARGETRANGE" not in array_cmds

    def test_range_uses_argetrange(self, redis_client, populated_array):
        calls: list[tuple] = []
        original = redis_client.execute_command

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        redis_client.execute_command = spy
        tools = build_tools(redis_client, _make_vectorizer(), _ARRAY_KEY, _INDEX_NAME)
        tools["fetch_lines"](start_line=1, end_line=3)

        array_cmds = [c[0] for c in calls]
        assert "ARGETRANGE" in array_cmds
        assert "ARGET" not in array_cmds

    def test_one_based_indexing_not_off_by_one(self, tools):
        """Line 1 must return the first element, not the second."""
        result = tools["fetch_lines"](start_line=1, end_line=1)
        assert "# Redis Persistence" in result
        # If off-by-one, this would return the blank line instead
        assert result.count("L1:") == 1

    def test_last_line_accessible(self, tools):
        result = tools["fetch_lines"](start_line=6, end_line=6)
        assert "L6: You can combine both strategies." in result

    def test_latency_present_in_observation(self, tools):
        result = tools["fetch_lines"](start_line=1, end_line=1)
        assert "latency:" in result
        assert "ms" in result


# ---------------------------------------------------------------------------
# argrep_search (ARGREP)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestArgrepSearchTool:
    def test_plain_text_matches_lines_containing_term(self, tools):
        result = tools["argrep_search"](pattern="AOF")
        # L3 "Redis supports RDB and AOF." and L5 "AOF logs every write command."
        assert "L3:" in result
        assert "L5:" in result

    def test_glob_heading_pattern(self, tools):
        result = tools["argrep_search"](pattern="# *")
        assert "L1: # Redis Persistence" in result

    def test_glob_no_match_double_hash(self, tools):
        """Fixture has no ## headings — must return a clear no-match message."""
        result = tools["argrep_search"](pattern="## *")
        assert "No lines matched" in result

    def test_regex_start_of_line(self, tools):
        result = tools["argrep_search"](pattern="^AOF")
        # Only L5 starts with AOF
        assert "L5:" in result
        assert "L3:" not in result

    def test_regex_pipe_alternation(self, tools):
        result = tools["argrep_search"](pattern="RDB|AOF")
        assert "L3:" in result
        assert "L4:" in result
        assert "L5:" in result

    def test_no_match_returns_descriptive_message(self, tools):
        result = tools["argrep_search"](pattern="NOTFOUND")
        assert "No lines matched" in result

    def test_zero_to_one_based_conversion(self, tools):
        """ARGREP returns 0-based indices; the tool must expose 1-based line numbers."""
        result = tools["argrep_search"](pattern="^# ")
        # First line is index 0 internally — must appear as L1 in output
        assert "L1:" in result
        assert "L0:" not in result

    def test_blank_lines_not_returned_by_grep(self, tools):
        """Blank lines should never match a non-empty pattern."""
        result = tools["argrep_search"](pattern="Redis")
        # L2 is blank — it must not appear in results
        assert "L2:" not in result

    def test_latency_present_in_observation(self, tools):
        result = tools["argrep_search"](pattern="RDB")
        assert "latency:" in result


# ---------------------------------------------------------------------------
# count_lines (ARLEN)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCountLinesTool:
    def test_returns_correct_line_count(self, tools):
        result = tools["count_lines"]()
        assert "6" in result  # fixture has exactly 6 lines

    def test_count_after_additional_insert(self, redis_client, populated_array):
        """ARLEN must reflect the current array length after new inserts."""
        redis_client.execute_command("ARINSERT", _ARRAY_KEY, "extra line")
        tools = build_tools(redis_client, _make_vectorizer(), _ARRAY_KEY, _INDEX_NAME)
        result = tools["count_lines"]()
        assert "7" in result

    def test_latency_present_in_observation(self, tools):
        result = tools["count_lines"]()
        assert "latency:" in result

    def test_calls_arlen_command(self, redis_client, populated_array):
        calls: list[tuple] = []
        original = redis_client.execute_command

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        redis_client.execute_command = spy
        tools = build_tools(redis_client, _make_vectorizer(), _ARRAY_KEY, _INDEX_NAME)
        tools["count_lines"]()

        assert any(c[0] == "ARLEN" for c in calls)
