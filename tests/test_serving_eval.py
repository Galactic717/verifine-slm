import asyncio
import json

import httpx
import pytest

from verifine.data.schemas import Case
from verifine.eval.stats import paired, summarize
from verifine.serve.client import Backend, Config
from verifine.serve.gateway import create_app


def test_stream_collects_content_usage_and_timings() -> None:
    body = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"{"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"}"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    async def run() -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text=body))
        async with Backend(Config(model="fake"), transport=transport) as model:
            row = await model.complete([{"role": "user", "content": "hello"}], stream=True)
        assert row.text == "{}"
        assert row.completion_tokens == 2
        assert row.ttft_ms is not None
        assert row.tpot_ms is not None

    asyncio.run(run())


def test_stream_rejects_missing_terminator() -> None:
    body = 'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\n'

    async def run() -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text=body))
        async with Backend(Config(model="fake"), transport=transport) as model:
            with pytest.raises(ValueError, match="incomplete generation"):
                await model.complete([{"role": "user", "content": "hello"}], stream=True)

    asyncio.run(run())


def test_gateway_validates_structured_response() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> None:
        app = create_app(Config(model="test"), httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/extract", json={"task": "invoice", "document": "A"}
                )
                assert response.status_code == 502
                assert "verifine_requests_total" in (await client.get("/metrics")).text

    asyncio.run(run())


def test_binary_metrics_and_paired_alignment() -> None:
    first = [
        {
            "id": "a",
            "language": "uk",
            "task": "invoice",
            "fields": {"x": "tp"},
            "json_valid": True,
            "schema_valid": True,
            "asserted": 1,
            "unsupported": 0,
            "elapsed_ms": 5,
            "exact": True,
            "error": None,
        }
    ]
    second = [{**first[0], "exact": False, "fields": {"x": "wrong"}}]
    assert summarize(first)["field_macro_f1"] == 1
    assert summarize(second)["field_macro_f1"] == 0
    assert paired(first, second, draws=20)["exact_delta"] == -1


def test_cloud_backend_cannot_send_unlabeled_batch() -> None:
    from verifine.eval.run_eval import check_data_boundary

    case = Case(
        id="x",
        group="manual",
        task="invoice",
        language="en",
        document="invoice",
        expected={
            "invoice_id": None,
            "supplier": None,
            "total_minor": None,
            "currency": None,
            "due_date": None,
        },
        synthetic=False,
    )
    try:
        check_data_boundary(
            [case],
            Config(kind="openai", url="https://api.openai.com/v1", model="baseline"),
            allow_remote=True,
        )
    except ValueError as error:
        assert "synthetic" in str(error)
    else:
        raise AssertionError("A non-synthetic payload escaped to a remote backend")
