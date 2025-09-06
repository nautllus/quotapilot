import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Ensure import path includes src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from router import Router, ProviderRegistry
from router.retry import calculate_backoff
from providers.base import ProviderAdapter
from gateway.schemas import ChatRequest, ChatResponse, ChatMessage, ChatChoice


class FakeBudget:
    def __init__(self):
        self.failures: List[Tuple[str, str, Optional[int]]] = []
        self.successes: List[Tuple[str, str, int, int]] = []

    async def record_usage(self, provider: str, model: str, request_tokens: int, response_tokens: int, success: bool, error_code: Optional[int] = None):
        if success:
            self.successes.append((provider, model, request_tokens, response_tokens))
        else:
            self.failures.append((provider, model, error_code))

    async def check_headroom(self, provider: str, model: str, est_prompt_tokens: Optional[int] = None, est_completion_tokens: Optional[int] = None):
        class HR:
            can_proceed = True
            remaining = {"rpm": None, "rpd": None, "tpm": None, "tpd": None}
        return HR()

    async def get_usage_stats(self, provider: str, model: str):
        return {"minute": {"requests": 0, "tokens": 0}, "day": {"requests": 0, "tokens": 0}}


class RLExc(Exception):
    def __init__(self, retry_after: Optional[str] = None):
        self.status_code = 429
        self.retry_after = retry_after


class HTTPExc(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code


class MockProvider(ProviderAdapter):
    def __init__(self, name: str):
        self._name = name
        self.calls = 0
        self.behavior: List[Any] = []  # sequence of results or exceptions

    @property
    def name(self) -> str:
        return self._name

    async def models(self) -> List[Dict[str, Any]]:
        return [{"name": f"{self._name}-model", "supports_json": True, "supports_stream": True, "supports_tools": True}]

    async def state(self) -> Dict[str, Any]:
        return {"status": "ok"}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.behavior:
            act = self.behavior.pop(0)
        else:
            act = HTTPExc(500)
        if isinstance(act, Exception):
            raise act
        # success path
        msg = ChatMessage(role="assistant", content=act)
        choice = ChatChoice(index=0, message=msg, finish_reason="stop")
        return ChatResponse(id="test", created=0, model=request.model, choices=[choice])


@pytest.mark.asyncio
async def test_retry_on_429_with_backoff(monkeypatch):
    # Speed up tests: patch sleep to no-op
    async def no_sleep(_):
        return None
    monkeypatch.setattr("router.core.asyncio.sleep", no_sleep)

    reg = ProviderRegistry()
    reg._providers = {}
    p = MockProvider("p1")
    p.behavior = [RLExc(retry_after="1"), "ok"]
    reg.register_provider(p)

    router = Router(registry=reg, budget=FakeBudget())
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], json=False)
    resp = await router.route_request(req)
    assert resp.choices[0].message.content == "ok"
    assert p.calls == 2


@pytest.mark.asyncio
async def test_failover_on_503_to_next_provider(monkeypatch):
    monkeypatch.setattr("router.core.asyncio.sleep", lambda _: None)

    reg = ProviderRegistry()
    reg._providers = {}
    p1 = MockProvider("p1")
    p1.behavior = [HTTPExc(503)]
    p2 = MockProvider("p2")
    p2.behavior = ["ok"]
    reg.register_provider(p1)
    reg.register_provider(p2)

    router = Router(registry=reg, budget=FakeBudget())
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    resp = await router.route_request(req)
    assert resp.choices[0].message.content == "ok"
    assert p1.calls == 1 and p2.calls == 1


@pytest.mark.asyncio
async def test_non_retryable_400_fails_immediately():
    reg = ProviderRegistry()
    reg._providers = {}
    p1 = MockProvider("p1")
    p1.behavior = [HTTPExc(400)]
    reg.register_provider(p1)

    router = Router(registry=reg, budget=FakeBudget())
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(HTTPExc):
        await router.route_request(req)


@pytest.mark.asyncio
async def test_max_retry_limit_enforced(monkeypatch):
    monkeypatch.setattr("router.core.asyncio.sleep", lambda _: None)

    reg = ProviderRegistry()
    reg._providers = {}
    p1 = MockProvider("p1")
    p1.behavior = [RLExc(), RLExc()]  # 2 attempts -> then give up, no other providers
    reg.register_provider(p1)

    router = Router(registry=reg, budget=FakeBudget())
    req = ChatRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(Exception):
        await router.route_request(req)
    assert p1.calls == 2

