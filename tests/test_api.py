from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.agent import TurnResult


def _make_turn_result(**kwargs) -> TurnResult:
    defaults = dict(
        user_message="test question",
        assistant_message="test answer",
        tool_used="grep",
        tool_reasoning="Exact search for a specific term.",
        grep_results=[{"line": 10, "content": "## Test Heading"}],
        total_latency_ms=3,
        vector_results=[],
        vector_latency_ms=None,
    )
    return TurnResult(**{**defaults, **kwargs})


@pytest.fixture
def client():
    mock_redis = MagicMock()
    mock_redis.exists.return_value = 1

    mock_executor = MagicMock()

    with (
        patch("backend.agent.get_redis_client", return_value=mock_redis),
        patch("backend.agent.ingest_document", return_value=("web:docs:test", "web:idx:test")),
        patch("backend.agent.document_ready", return_value=True),
        patch("backend.agent.build_executor", return_value=mock_executor),
        patch("backend.agent.run_turn", return_value=_make_turn_result()),
        patch("backend.agent.load_config") as mock_cfg,
        patch("redisvl.utils.vectorize.OpenAITextVectorizer"),
    ):
        mock_cfg.return_value = MagicMock(
            redis_url="redis://localhost:6379",
            openai_api_key="sk-test",
            openai_model="gpt-4.1-mini",
            markdown_file="./docs/test.md",
        )

        from backend.app import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestReady:
    def test_ready_returns_ok_when_document_ready(self, client):
        resp = client.get("/api/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "loading")
        assert "array_key" in data
        assert "index_name" in data
        assert "document_ready" in data


class TestChat:
    def test_chat_rejects_empty_message(self, client):
        resp = client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_chat_returns_expected_fields(self, client):
        resp = client.post("/api/chat", json={"message": "find all headings"})
        assert resp.status_code == 200
        data = resp.json()
        assert "user_message" in data
        assert "assistant_message" in data
        assert "tool_used" in data
        assert "tool_reasoning" in data
        assert "grep_results" in data
        assert "vector_results" in data
        assert "total_latency_ms" in data
        assert "vector_latency_ms" in data

    def test_chat_tool_used_is_valid_value(self, client):
        resp = client.post("/api/chat", json={"message": "show me line 5"})
        assert resp.status_code == 200
        assert resp.json()["tool_used"] in ("grep", "vector", "fetch", "both", "none")

    def test_chat_grep_results_have_line_and_content(self, client):
        resp = client.post("/api/chat", json={"message": "find all headings"})
        assert resp.status_code == 200
        for r in resp.json()["grep_results"]:
            assert "line" in r
            assert "content" in r
