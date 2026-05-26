from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redisvl.utils.vectorize import OpenAITextVectorizer
from pydantic import BaseModel, Field

from .agent import (
    AgentExecutor,
    DemoConfig,
    TurnResult,
    build_executor,
    document_ready,
    docs_key,
    get_redis_client,
    idx_key,
    ingest_document,
    load_config,
    run_turn,
)

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# App-level singletons — populated during lifespan startup
# ---------------------------------------------------------------------------

_config: DemoConfig | None = None
_array_key: str | None = None
_index_name: str | None = None
_executor: AgentExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _array_key, _index_name, _executor

    _config = load_config()
    redis_client = get_redis_client(_config.redis_url)
    vectorizer = OpenAITextVectorizer(
        model="text-embedding-3-small",
        api_config={"api_key": _config.openai_api_key},
    )

    _array_key, _index_name = ingest_document(
        redis_client,
        vectorizer,
        _config.markdown_file,
    )

    _executor = build_executor(_config, redis_client, _array_key, _index_name)

    logger.info(
        "Startup complete. Array key: %s  Vector index: %s",
        _array_key,
        _index_name,
    )
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Redis Array Context Demo",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class GrepResult(BaseModel):
    line: int
    content: str
    latency_ms: float | None = None


class VectorResult(BaseModel):
    score: float
    content: str


class ChatResponse(BaseModel):
    user_message: str
    assistant_message: str
    tool_used: str
    tool_reasoning: str
    tool_commands: list[str]
    grep_results: list[GrepResult]
    grep_latency_ms: float | None
    vector_results: list[VectorResult]
    vector_latency_ms: float | None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    array_key: str
    index_name: str
    document_ready: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    if _config is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    redis_client = get_redis_client(_config.redis_url)
    ready = document_ready(redis_client, _config.markdown_file)

    return ReadinessResponse(
        status="ok" if ready else "loading",
        array_key=_array_key or "",
        index_name=_index_name or "",
        document_ready=ready,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if _executor is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    try:
        result: TurnResult = run_turn(_executor, request.message.strip())
    except Exception as exc:
        logger.exception("Agent turn failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        user_message=result.user_message,
        assistant_message=result.assistant_message,
        tool_used=result.tool_used,
        tool_reasoning=result.tool_reasoning,
        tool_commands=result.tool_commands,
        grep_results=[GrepResult(**r) for r in result.grep_results],
        grep_latency_ms=result.grep_latency_ms,
        vector_results=[VectorResult(**r) for r in result.vector_results],
        vector_latency_ms=result.vector_latency_ms,
    )
