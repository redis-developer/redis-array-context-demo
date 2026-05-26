from __future__ import annotations

import pytest
from testcontainers.redis import RedisContainer

# Match the exact image used in docker-compose.yml so integration tests
# exercise the same Redis version as production.
_REDIS_IMAGE = "redis:8.8.0"


@pytest.fixture(scope="session")
def redis_container():
    """
    Spin up a single Redis 8.8-rc1 container for the whole test session.
    Session scope avoids the ~2s Docker startup cost on every test.
    """
    with RedisContainer(image=_REDIS_IMAGE) as container:
        yield container


@pytest.fixture()
def redis_client(redis_container):
    """
    Return a decode_responses=True Redis client, then flush all keys after
    each test so tests are fully isolated without restarting the container.
    """
    client = redis_container.get_client(decode_responses=True)
    yield client
    client.flushall()
