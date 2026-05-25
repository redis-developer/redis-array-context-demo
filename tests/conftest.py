from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_redis():
    """A mock Redis client that returns sensible defaults."""
    client = MagicMock()
    client.exists.return_value = 1
    client.execute_command.return_value = []
    return client


@pytest.fixture
def mock_agent_executor():
    """A mock LangChain AgentExecutor."""
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "This is a test answer.",
        "intermediate_steps": [],
    }
    return executor


@pytest.fixture
def client(mock_redis, mock_agent_executor):
    """FastAPI TestClient with Redis and agent mocked out."""
    with (
        patch("backend.agent.get_redis_client", return_value=mock_redis),
        patch("backend.agent.build_executor", return_value=mock_agent_executor),
    ):
        from backend.app import app
        yield TestClient(app)
