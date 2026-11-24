import asyncio

import httpx

from verifine.serve.client import Config
from verifine.serve.gateway import create_app


def test_chat_preserves_upstream_rate_limit_status_and_delay() -> None:
    upstream = httpx.MockTransport(
        lambda _: httpx.Response(429, headers={"Retry-After": "30"}, json={"message": "internal"})
    )

    async def run() -> None:
        app = create_app(Config(model="test"), upstream)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "example"}]},
                )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "30"
        assert response.json()["error"]["type"] == "rate_limit_error"
        assert "internal" not in response.text

    asyncio.run(run())
