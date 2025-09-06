import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Ensure import path includes src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gateway.main import app
from gateway.schemas import ChatRequest, ChatResponse, ChatMessage, ChatChoice


class _FakeRouter:
    async def route_request(self, req: ChatRequest) -> ChatResponse:
        msg = ChatMessage(role="assistant", content="hello")
        choice = ChatChoice(index=0, message=msg, finish_reason="stop")
        return ChatResponse(id="x", created=0, model=req.model if req.model else "auto", choices=[choice])


def _with_fake_router():
    app.state.router = _FakeRouter()  # type: ignore[attr-defined]
    return TestClient(app)


def test_chat_completions_basic_ok():
    client = _with_fake_router()
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hi"}],
    }
    res = client.post("/v1/chat/completions", json=body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["choices"][0]["message"]["content"] == "hello"


def test_chat_completions_streaming_sse():
    client = _with_fake_router()
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    res = client.post("/v1/chat/completions", json=body)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")
    text = res.text
    assert "[DONE]" in text


def test_chat_completions_validation_error():
    client = _with_fake_router()
    # missing required fields -> FastAPI/Pydantic will reject with 422
    res = client.post("/v1/chat/completions", json={"model": "auto"})
    assert res.status_code in (400, 422)

