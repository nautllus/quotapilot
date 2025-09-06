import json
from typing import Any, Dict, List

import pytest

# Ensure import path includes src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from router import ProviderRegistry, Router, NoCapableProviderError
from providers.base import ProviderAdapter
from gateway.schemas import ChatRequest, ChatResponse, ChatMessage, ChatChoice


class MockProvider(ProviderAdapter):
    def __init__(self, name: str, models: List[Dict[str, Any]], response_text: str = "ok") -> None:
        self._name = name
        self._models = models
        self._response_text = response_text

    @property
    def name(self) -> str:
        return self._name

    async def models(self) -> List[Dict[str, Any]]:
        return self._models

    async def state(self) -> Dict[str, Any]:
        return {"status": "ok"}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        msg = ChatMessage(role="assistant", content=self._response_text)
        choice = ChatChoice(index=0, message=msg, finish_reason="stop")
        return ChatResponse(id="test", created=0, model=request.model, choices=[choice])


@pytest.mark.asyncio
async def test_registry_registration_and_retrieval():
    reg = ProviderRegistry()
    p = MockProvider("mock", models=[])
    reg.register_provider(p)
    assert reg.get_provider("mock") is p
    assert p in reg.get_providers()


@pytest.mark.asyncio
async def test_router_filters_by_json_and_stream_and_tools():
    reg = ProviderRegistry()
    # Clear auto-registered providers for isolation (create fresh registry without env)
    reg._providers = {}

    p1 = MockProvider("p1", models=[{"name": "m1", "supports_json": True, "supports_stream": False, "supports_tools": True}])
    p2 = MockProvider("p2", models=[{"name": "m2", "supports_json": True, "supports_stream": True, "supports_tools": True}])
    reg.register_provider(p1)
    reg.register_provider(p2)

    router = Router(reg)

    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], json=True, stream=True, tools=[{"type": "function"}])
    resp = await router.route_request(req)
    # Should select p2.m2 (only one that supports stream True among those that also support json and tools)
    assert resp.model == "m2"


@pytest.mark.asyncio
async def test_router_honors_provider_hint():
    reg = ProviderRegistry()
    reg._providers = {}
    p1 = MockProvider("p1", models=[{"name": "alpha", "supports_json": True, "supports_stream": True, "supports_tools": True}])
    p2 = MockProvider("p2", models=[{"name": "beta", "supports_json": True, "supports_stream": True, "supports_tools": True}])
    reg.register_provider(p1)
    reg.register_provider(p2)

    router = Router(reg)
    # provider hint
    req = ChatRequest(model="p1:alpha", messages=[ChatMessage(role="user", content="hi")])
    resp = await router.route_request(req)
    assert resp.model == "alpha"


@pytest.mark.asyncio
async def test_router_no_capable_provider_raises():
    reg = ProviderRegistry()
    reg._providers = {}
    p1 = MockProvider("p1", models=[{"name": "m1", "supports_json": False, "supports_stream": False, "supports_tools": False}])
    reg.register_provider(p1)

    router = Router(reg)
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], json=True)
    with pytest.raises(NoCapableProviderError):
        await router.route_request(req)

