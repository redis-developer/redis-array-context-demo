from __future__ import annotations

import pytest
from testcontainers.redis import RedisContainer

# Match the exact image used in docker-compose.yml so integration tests
# exercise the same Redis version as production.
_REDIS_IMAGE = "redis:8.8.0"

# Port used by the docker-compose Redis service.  Integration tests must NEVER
# connect here — that would silently corrupt application data.
_COMPOSE_REDIS_PORT = 6379


@pytest.fixture(scope="session")
def redis_container():
    """
    Spin up a dedicated Redis 8.8.0 container for the whole test session.
    Session scope avoids the container startup cost on every test.

    The container ID and mapped port are printed so you can verify a fresh
    container is started (not the docker-compose Redis on port 6379).
    """
    with RedisContainer(image=_REDIS_IMAGE) as container:
        port = int(container.get_exposed_port(_COMPOSE_REDIS_PORT))
        container_id = container.get_wrapped_container().short_id
        print(
            f"\n[testcontainers] Redis {_REDIS_IMAGE} started — "
            f"container {container_id}, host port {port}"
        )
        if port == _COMPOSE_REDIS_PORT:
            pytest.fail(
                f"testcontainers returned port {port} — the tests would run "
                "against the docker-compose Redis and destroy application data. "
                "Ensure Docker is available so testcontainers can allocate a "
                "fresh container on a random port."
            )
        yield container
        print(f"\n[testcontainers] Stopping container {container_id}")


@pytest.fixture()
def redis_client(redis_container):
    """
    Return a decode_responses=True Redis client pointing at the test container.
    Keys are flushed before AND after each test so tests are fully isolated
    even if a previous session left data behind.
    """
    client = redis_container.get_client(decode_responses=True)
    client.flushall()   # clean slate at the START
    yield client
    client.flushall()   # clean up at the END
