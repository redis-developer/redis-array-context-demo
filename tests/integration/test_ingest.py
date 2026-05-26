"""
Integration tests for ingest_document against a real Redis 8.8-rc1 container.

The OpenAI vectorizer is mocked — these tests focus on whether the Array
ingestion and index creation work correctly against the actual Redis commands,
not on embedding quality.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agent import CLI_PREFIX, WEB_PREFIX, docs_key, idx_key, ingest_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vectorizer(dim: int = 1536) -> MagicMock:
    """
    Mock vectorizer whose embed_many returns one float vector per input text,
    matching the real OpenAI text-embedding-3-small output shape.
    """
    v = MagicMock()
    v.embed.return_value = [0.1] * dim
    v.embed_many.side_effect = lambda texts, **_: [[0.1] * dim for _ in texts]
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIngestDocument:
    def test_array_key_created(self, redis_client, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("Line one\nLine two\nLine three\n")

        array_key, _ = ingest_document(redis_client, _make_vectorizer(), str(md))

        assert redis_client.exists(array_key) == 1

    def test_line_count_matches_file(self, redis_client, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("Line one\nLine two\nLine three\n")

        array_key, _ = ingest_document(redis_client, _make_vectorizer(), str(md))

        assert redis_client.execute_command("ARLEN", array_key) == 3

    def test_line_content_retrievable_via_arget(self, redis_client, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Heading\nSome content\n")

        array_key, _ = ingest_document(redis_client, _make_vectorizer(), str(md))

        # ARGET is 0-based; line 1 → index 0, line 2 → index 1
        assert redis_client.execute_command("ARGET", array_key, 0) == "# Heading"
        assert redis_client.execute_command("ARGET", array_key, 1) == "Some content"

    def test_blank_lines_preserved(self, redis_client, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("First\n\nThird\n")

        array_key, _ = ingest_document(redis_client, _make_vectorizer(), str(md))

        # Blank line must be stored, not skipped, so line positions stay stable
        assert redis_client.execute_command("ARLEN", array_key) == 3
        assert redis_client.execute_command("ARGET", array_key, 1) == ""

    def test_idempotent_on_second_call(self, redis_client, tmp_path):
        """Calling ingest_document twice on the same key must not duplicate lines."""
        md = tmp_path / "doc.md"
        md.write_text("Line A\nLine B\n")

        key1, idx1 = ingest_document(redis_client, _make_vectorizer(), str(md))
        key2, idx2 = ingest_document(redis_client, _make_vectorizer(), str(md))

        assert key1 == key2
        assert idx1 == idx2
        assert redis_client.execute_command("ARLEN", key1) == 2  # not 4

    def test_key_uses_web_prefix_by_default(self, redis_client, tmp_path):
        md = tmp_path / "redis-arrays.md"
        md.write_text("content\n")

        array_key, index_name = ingest_document(redis_client, _make_vectorizer(), str(md))

        assert array_key == docs_key(str(md), WEB_PREFIX)
        assert index_name == idx_key(str(md), WEB_PREFIX)
        assert array_key.startswith("web:")

    def test_key_uses_cli_prefix_when_specified(self, redis_client, tmp_path):
        md = tmp_path / "redis-arrays.md"
        md.write_text("content\n")

        array_key, index_name = ingest_document(
            redis_client, _make_vectorizer(), str(md), prefix=CLI_PREFIX
        )

        assert array_key.startswith("cli:")
        assert index_name.startswith("cli:")

    def test_vector_index_created(self, redis_client, tmp_path):
        """The FT index must exist and be queryable after ingestion."""
        md = tmp_path / "doc.md"
        md.write_text("Redis supports AOF persistence.\nRDB is snapshot-based.\n")

        _, index_name = ingest_document(redis_client, _make_vectorizer(), str(md))

        # FT.INFO raises an error if the index doesn't exist
        info = redis_client.execute_command("FT.INFO", index_name)
        assert info is not None
