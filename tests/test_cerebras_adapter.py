import json
from typing import Any, Dict, List

import httpx
import pytest

# Ensure import path includes src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from providers.cerebras import CerebrasAdapter
from providers.base_openai import BaseRateLimitError
from gateway.schemas import ChatRequest, ChatMessage


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handlers: Dict[str, Any]):
        self.handlers = handlers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        key = f"{request.method} {request.url.path}"
        handler = self.handlers.get(key)
        if handler is None:
            return httpx.Response(404, request=request, json={"error": "not found"})
        return await handler(request)


@pytest.mark.asyncio
async def test_cerebras_models_from_config():
    adapter = CerebrasAdapter(models_config=[
        {"name": "llama3.1-8b", "supports_json": True, "supports_tools": False, "supports_stream": True, "context_window": 8192}
    ])
    models = await adapter.models()
    assert models[0]["name"] == "llama3.1-8b"
    assert models[0]["supports_stream"] is True
    assert models[0]["supports_tools"] is False


@pytest.mark.asyncio
async def test_cerebras_chat_completion_mapping(monkeypatch):
    async def chat_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "llama3.1-8b"
        assert payload["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "id": "cmpl-abc",
                "object": "chat.completion",
                "created": 123,
                "model": "llama3.1-8b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    transport = _MockTransport({
        "POST /v1/chat/completions": chat_handler,
    })
    client = httpx.AsyncClient(base_url="https://api.cerebras.ai/v1", transport=transport)

    adapter = CerebrasAdapter(client=client, models_config=[{"name": "llama3.1-8b"}])

    req = ChatRequest(model="llama3.1-8b", messages=[ChatMessage(role="user", content="hello")])
    resp = await adapter.chat(req)
    assert resp.id == "cmpl-abc"
    assert resp.usage.total_tokens == 6

    await client.aclose()


@pytest.mark.asyncio
async def test_cerebras_rate_limit_error(monkeypatch):
    async def rl_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "5"}, json={"error": "rate limited"})

    transport = _MockTransport({
        "POST /v1/chat/completions": rl_handler,
    })
    client = httpx.AsyncClient(base_url="https://api.cerebras.ai/v1", transport=transport)

    adapter = CerebrasAdapter(client=client, models_config=[{"name": "llama3.1-8b"}])
    req = ChatRequest(model="llama3.1-8b", messages=[ChatMessage(role="user", content="hi")])

    with pytest.raises(BaseRateLimitError) as exc:
        await adapter.chat(req)
    assert exc.value.status_code == 429
    assert exc.value.retry_after == "5"

    await client.aclose()

