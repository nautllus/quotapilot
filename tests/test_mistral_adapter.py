import asyncio
import json
import os
from typing import Any, Dict

import httpx
import pytest

# Ensure 'src' is importable
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from providers.mistral import MistralAdapter, MistralRateLimitError
from gateway.schemas import ChatRequest, ChatMessage


class _MockTransport(httpx.AsyncBaseTransport):
    """A simple mock transport for httpx.AsyncClient."""

    def __init__(self, handlers: Dict[str, Any]):
        self.handlers = handlers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        key = f"{request.method} {request.url.path}"
        handler = self.handlers.get(key)
        if handler is None:
            return httpx.Response(404, request=request, json={"error": "not found"})
        return await handler(request)


@pytest.mark.asyncio
async def test_models_free_tier_filtering(monkeypatch):
    # Prepare mock /v1/models
    async def models_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "mistral-tiny"},
                    {"id": "open-mixtral-8x7b"},
                    {"id": "mistral-large-latest"},
                ]
            },
        )

    transport = _MockTransport({
        "GET /v1/models": models_handler,
    })

    client = httpx.AsyncClient(base_url="https://api.mistral.ai/v1", transport=transport)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_FREE_MODELS", "mistral-tiny,open-mixtral-8x7b")

    adapter = MistralAdapter(client=client)
    models = await adapter.models()

    names = [m["name"] for m in models]
    assert names == ["mistral-tiny", "open-mixtral-8x7b"]
    for m in models:
        assert m["supports_stream"] is True
        assert m["supports_json"] is True
        assert m["supports_tools"] is True

    await client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_mapping(monkeypatch):
    # Mock /v1/chat/completions
    async def chat_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "mistral-tiny"
        assert payload["messages"][0]["role"] == "user"
        assert payload.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "id": "cmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mistral-tiny",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    transport = _MockTransport({
        "POST /v1/chat/completions": chat_handler,
    })

    client = httpx.AsyncClient(base_url="https://api.mistral.ai/v1", transport=transport)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    adapter = MistralAdapter(client=client)

    req = ChatRequest(
        model="mistral-tiny",
        messages=[ChatMessage(role="user", content="hello")],
        json=True,
    )

    resp = await adapter.chat(req)

    assert resp.id == "cmpl-123"
    assert resp.model == "mistral-tiny"
    assert resp.choices[0].message.content == "hi"
    assert resp.usage.total_tokens == 7

    await client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_error_raised(monkeypatch):
    async def chat_rl_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "10"}, json={"error": "rate limited"})

    transport = _MockTransport({
        "POST /v1/chat/completions": chat_rl_handler,
    })

    client = httpx.AsyncClient(base_url="https://api.mistral.ai/v1", transport=transport)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    adapter = MistralAdapter(client=client)
    req = ChatRequest(model="mistral-tiny", messages=[ChatMessage(role="user", content="hi")])

    with pytest.raises(MistralRateLimitError) as exc:
        await adapter.chat(req)
    assert exc.value.status_code == 429
    assert exc.value.retry_after == "10"

    await client.aclose()

