import hmac
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from verifine.data.prompts import messages
from verifine.data.schemas import SCHEMAS, Task
from verifine.rewards.schema_reward import parse_output
from verifine.serve.client import Backend, Config, from_env, token_usage


class Extract(BaseModel):
    task: Task
    document: Annotated[str, Field(min_length=1, max_length=16000)]
    constrained: bool = True


def create_app(
    config: Config | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = config or from_env()
    key = os.getenv("VERIFINE_GATEWAY_KEY", "")
    registry = CollectorRegistry()
    requests = Counter(
        "verifine_requests", "Completed requests", ["route", "status"], registry=registry
    )
    latency = Histogram(
        "verifine_request_seconds", "Request duration", ["route"], registry=registry
    )
    tokens = Counter(
        "verifine_tokens", "Backend-reported token count", ["direction"], registry=registry
    )
    cost = Counter(
        "verifine_estimated_cost_usd", "Configured token price estimate", registry=registry
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with Backend(cfg, transport=transport) as upstream:
            app.state.upstream = upstream
            yield

    app = FastAPI(title="Verifine SLM", lifespan=lifespan)
    if cfg.kind in {"openai", "anthropic"} or cfg.host not in {"localhost", "127.0.0.1", "::1"}:
        if os.getenv("VERIFINE_ALLOW_NETWORK_BACKEND") != "1":
            raise ValueError("Set VERIFINE_ALLOW_NETWORK_BACKEND=1 for a network upstream")

    def authenticate(authorization: str | None) -> None:
        if key and not hmac.compare_digest(
            (authorization or "").encode(),
            ("Bearer " + key).encode(),
        ):
            raise HTTPException(401, "Invalid gateway credentials")

    def meter(
        route: str,
        status: str,
        started: float,
        prompt: int | None = None,
        output: int | None = None,
    ) -> None:
        requests.labels(route, status).inc()
        latency.labels(route).observe(perf_counter() - started)
        if prompt is not None:
            tokens.labels("input").inc(prompt)
        if output is not None:
            tokens.labels("output").inc(output)
        estimate = cfg.price(prompt, output)
        if estimate is not None:
            cost.inc(estimate)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        authenticate(authorization)
        return {
            "object": "list",
            "data": [{"id": cfg.model, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/extract")
    async def extract(
        request: Extract,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        authenticate(authorization)
        started = perf_counter()
        route = "extract"
        upstream: Backend = app.state.upstream
        try:
            answer = await upstream.complete(
                messages(request.task, request.document),
                schema=SCHEMAS[request.task].model_json_schema() if request.constrained else None,
            )
        except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError):
            meter(route, "upstream_error", started)
            raise HTTPException(502, "Backend could not complete the request") from None
        _, parsed = parse_output(request.task, answer.text)
        meter(
            route,
            "ok" if parsed else "invalid_output",
            started,
            answer.prompt_tokens,
            answer.completion_tokens,
        )
        if parsed is None:
            raise HTTPException(502, "Backend returned invalid structured output")
        return {
            "model": cfg.model,
            "task": request.task,
            "result": parsed.model_dump(),
            "usage": {
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat(
        payload: dict[str, Any],
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        authenticate(authorization)
        route = "chat"
        if not isinstance(payload.get("messages"), list) or len(json.dumps(payload)) > 120_000:
            raise HTTPException(422, "Expected messages in a request of at most 120,000 characters")
        started = perf_counter()
        streamed = payload.get("stream") is True
        payload["model"] = cfg.model
        if streamed:
            payload["stream_options"] = {"include_usage": True}
        try:
            upstream: Backend = app.state.upstream
            remote = await upstream.raw_chat(payload)
        except (httpx.HTTPError, ValueError):
            meter(route, "upstream_error", started)
            raise HTTPException(502, "Backend connection failed") from None
        if remote.is_error:
            await remote.aclose()
            meter(route, "upstream_error", started)
            status = remote.status_code if remote.is_client_error else 502
            error_type = (
                "rate_limit_error"
                if status == 429
                else "invalid_request_error"
                if status < 500
                else "server_error"
            )
            headers = (
                {"Retry-After": remote.headers["Retry-After"]}
                if (status == 429 and "Retry-After" in remote.headers)
                else None
            )
            return JSONResponse(
                {
                    "error": {
                        "message": "Backend rejected the request",
                        "type": error_type,
                        "param": None,
                        "code": None,
                    }
                },
                status_code=status,
                headers=headers,
            )
        if streamed:

            async def forward() -> AsyncIterator[str]:
                prompt = output = None
                status = "interrupted"
                try:
                    async for line in remote.aiter_lines():
                        if line.startswith("data:"):
                            packet = line[5:].strip()
                            if packet == "[DONE]":
                                status = "ok"
                            else:
                                chunk = json.loads(packet)
                                if chunk.get("usage"):
                                    prompt, output = token_usage(chunk)
                        yield line + "\n"
                finally:
                    await remote.aclose()
                    meter(route, status, started, prompt, output)

            return StreamingResponse(forward(), media_type="text/event-stream")
        try:
            await remote.aread()
            result = remote.json()
            count = token_usage(result)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            meter(route, "upstream_error", started)
            raise HTTPException(502, "Backend returned invalid JSON") from None
        finally:
            await remote.aclose()
        meter(route, "ok", started, *count)
        return JSONResponse(result)

    return app
