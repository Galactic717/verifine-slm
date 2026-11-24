import json
import math
import os
from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter
from typing import Any, Literal, cast

import httpx

from verifine.data.prompts import Message

Kind = Literal["llama", "vllm", "sglang", "ollama", "openai", "anthropic"]
KINDS = ("llama", "vllm", "sglang", "ollama", "openai", "anthropic")


@dataclass(frozen=True)
class Config:
    model: str = "gemma-4-E4B-local"
    url: str = "http://127.0.0.1:9940/v1"
    kind: Kind = "llama"
    api_key: str = field(default="", repr=False)
    timeout: float = 90.0
    input_usd: float | None = None
    output_usd: float | None = None

    def __post_init__(self) -> None:
        url = httpx.URL(self.url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise ValueError("Use an HTTP(S) URL without credentials")
        if url.query or url.fragment or not self.model.strip() or self.kind not in KINDS:
            raise ValueError("Invalid backend URL, model, or kind")
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 600:
            raise ValueError("Timeout must be between 0 and 600 seconds")
        for rate in (self.input_usd, self.output_usd):
            if rate is not None and (not math.isfinite(rate) or rate < 0):
                raise ValueError("Token prices must be finite and non-negative")

    @property
    def host(self) -> str:
        return httpx.URL(self.url).host

    @property
    def base(self) -> str:
        return self.url.rstrip("/").removesuffix("/v1")

    def price(self, prompt: int | None, output: int | None) -> float | None:
        if prompt is None or output is None or self.input_usd is None or self.output_usd is None:
            return None
        return (prompt * self.input_usd + output * self.output_usd) / 1e6


def from_env() -> Config:
    def price(name: str) -> float | None:
        value = os.getenv(name)
        return float(value) if value else None

    return Config(
        model=os.getenv("VERIFINE_MODEL", Config.model),
        url=os.getenv("VERIFINE_BACKEND_URL", Config.url),
        kind=cast(Kind, os.getenv("VERIFINE_BACKEND", "llama")),
        api_key=os.getenv("VERIFINE_API_KEY", ""),
        timeout=float(os.getenv("VERIFINE_TIMEOUT", str(Config.timeout))),
        input_usd=price("VERIFINE_INPUT_USD_PER_MTOK"),
        output_usd=price("VERIFINE_OUTPUT_USD_PER_MTOK"),
    )


@dataclass(frozen=True)
class Completion:
    text: str
    elapsed_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    answer_ttft_ms: float | None = None
    confidence: float | None = None


def token_usage(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


class Backend:
    def __init__(self, config: Config, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.http = httpx.AsyncClient(
            timeout=config.timeout,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )
        self.headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}

    async def __aenter__(self) -> "Backend":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.http.aclose()

    async def raw_chat(self, payload: dict[str, Any]) -> httpx.Response:
        if self.config.kind == "anthropic":
            raise ValueError("Anthropic uses the evaluation client only")
        request = self.http.build_request(
            "POST",
            self.config.base + "/v1/chat/completions",
            json=payload,
            headers=self.headers,
        )
        return await self.http.send(request, stream=True)

    async def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 240,
        seed: int = 42,
        logprobs: bool = False,
    ) -> Completion:
        started = perf_counter()
        if logprobs and (stream or self.config.kind in {"ollama", "anthropic"}):
            raise ValueError("Confidence requires non-streamed OpenAI-compatible token logprobs")
        if self.config.kind == "ollama":
            return await self._ollama(messages, schema, stream, max_tokens, seed, started)
        if self.config.kind == "anthropic":
            if stream or schema:
                raise ValueError("Anthropic baseline supports unassisted quality evaluation")
            return await self._anthropic(messages, max_tokens, started)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if self.config.kind != "openai":
            payload["seed"] = seed
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if logprobs:
            payload["logprobs"] = True
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema},
            }
        response = await self.raw_chat(payload)
        try:
            response.raise_for_status()
            if stream:
                return await self._stream(response, started, native_ollama=False)
            await response.aread()
            result = response.json()
            choice = result["choices"][0]
            scores = (choice.get("logprobs") or {}).get("content") or []
            log_values = [
                value
                for item in scores
                if isinstance(value := item.get("logprob"), int | float)
                and math.isfinite(value)
                and value <= 0
            ]
            return Completion(
                choice["message"].get("content") or "",
                (perf_counter() - started) * 1000,
                None,
                None,
                *token_usage(result),
                choice.get("finish_reason"),
                confidence=math.exp(mean(log_values)) if log_values else None,
            )
        finally:
            await response.aclose()

    async def _stream(
        self,
        response: httpx.Response,
        started: float,
        native_ollama: bool,
    ) -> Completion:
        text: list[str] = []
        prompt = output = None
        first = last = None
        first_answer = None
        token_chunks = 0
        finish = None
        done = False
        async for line in response.aiter_lines():
            if native_ollama:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise ValueError("Ollama streaming error")
                delta = chunk.get("message", {})
                content = delta.get("content")
                reasoning = delta.get("thinking")
                if chunk.get("done"):
                    done = True
                    prompt, output = chunk.get("prompt_eval_count"), chunk.get("eval_count")
                    finish = chunk.get("done_reason")
            else:
                if not line.startswith("data:"):
                    continue
                packet = line[5:].strip()
                if packet == "[DONE]":
                    done = True
                    break
                chunk = json.loads(packet)
                if "error" in chunk:
                    raise ValueError("Backend streaming error")
                choices = chunk.get("choices") or [{}]
                finish = choices[0].get("finish_reason") or finish
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if chunk.get("usage"):
                    prompt, output = token_usage(chunk)
            if content or reasoning:
                token_chunks += 1
                last = perf_counter()
                if first is None:
                    first = last
            if content:
                if first_answer is None:
                    first_answer = last
                text.append(content)
        if not done:
            raise ValueError("Backend closed an incomplete generation stream")
        ttft = (first - started) * 1000 if first is not None else None
        tpot = (
            (last - first) * 1000 / (output - 1)
            if last is not None and first is not None and output and output > 1 and token_chunks > 1
            else None
        )
        return Completion(
            "".join(text),
            (perf_counter() - started) * 1000,
            ttft,
            tpot,
            prompt,
            output,
            finish,
            answer_ttft_ms=(first_answer - started) * 1000 if first_answer is not None else None,
        )

    async def _ollama(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None,
        stream: bool,
        max_tokens: int,
        seed: int,
        started: float,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens},
        }
        if schema:
            payload["format"] = schema
        async with self.http.stream(
            "POST", self.config.base + "/api/chat", json=payload, headers=self.headers
        ) as response:
            response.raise_for_status()
            if stream:
                return await self._stream(response, started, native_ollama=True)
            await response.aread()
            data = response.json()
        return Completion(
            data["message"]["content"],
            (perf_counter() - started) * 1000,
            None,
            None,
            data.get("prompt_eval_count"),
            data.get("eval_count"),
            data.get("done_reason"),
        )

    async def _anthropic(
        self,
        messages: list[Message],
        max_tokens: int,
        started: float,
    ) -> Completion:
        response = await self.http.post(
            self.config.base + "/v1/messages",
            headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.config.model,
                "system": messages[0]["content"],
                "messages": messages[1:],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        content = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        usage = data.get("usage") or {}
        return Completion(
            content,
            (perf_counter() - started) * 1000,
            None,
            None,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            data.get("stop_reason"),
        )
