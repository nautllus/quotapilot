import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

# Ensure import path includes src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from state.budget import BudgetManager


class FakeCursor:
    def __init__(self, docs: List[Dict[str, Any]]):
        self._docs = docs

    async def to_list(self, length: Optional[int] = None):
        return self._docs[: length or len(self._docs)]


class FakeCollection:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    async def insert_one(self, doc: Dict[str, Any]):
        self.docs.append(doc)

    def aggregate(self, pipeline, allowDiskUse=False):  # noqa: N803
        # Very small interpreter for our expected pipeline
        # pipeline[0] = {"$match": {"provider":..., "model":..., "ts": {"$gte": start}}}
        # pipeline[1] = {"$group": {"_id": None, "requests": {"$sum": 1}, "tokens": {"$sum": "$total_tokens"}}}
        match = pipeline[0]["$match"]
        provider = match.get("provider")
        model = match.get("model")
        start = match.get("ts", {}).get("$gte")
        filtered = [d for d in self.docs if d["provider"] == provider and d["model"] == model and d["ts"] >= start]
        requests = len(filtered)
        tokens = sum(int(d.get("total_tokens", 0) or 0) for d in filtered)
        return FakeCursor([{"_id": None, "requests": requests, "tokens": tokens}])


@pytest.mark.asyncio
async def test_record_usage_and_stats():
    col = FakeCollection()
    budget = BudgetManager(collection=col, limits={})

    now = datetime.utcnow()
    await budget.record_usage("mistral", "mistral-tiny", request_tokens=10, response_tokens=20, ts=now)
    await budget.record_usage("mistral", "mistral-tiny", request_tokens=5, response_tokens=5, ts=now - timedelta(seconds=30))
    # Outside window
    await budget.record_usage("mistral", "mistral-tiny", request_tokens=1, response_tokens=1, ts=now - timedelta(minutes=2))

    stats = await budget.get_usage_stats("mistral", "mistral-tiny")
    assert stats["minute"]["requests"] == 2
    assert stats["minute"]["tokens"] == 40
    assert stats["day"]["requests"] == 3


@pytest.mark.asyncio
async def test_headroom_checks():
    col = FakeCollection()
    limits = {
        "mistral": {
            "mistral-tiny": {"rpm": 2, "rpd": 5, "tpm": 100, "tpd": 1000}
        }
    }
    budget = BudgetManager(collection=col, limits=limits)

    now = datetime.utcnow()
    # Use up some tokens/requests
    await budget.record_usage("mistral", "mistral-tiny", request_tokens=40, response_tokens=0, ts=now)

    # Headroom should allow one more request and limited tokens
    hr = await budget.check_headroom("mistral", "mistral-tiny", est_prompt_tokens=30, est_completion_tokens=20)
    assert hr.can_proceed is True
    assert hr.remaining["rpm"] == 1

    # Exceed tpm
    hr2 = await budget.check_headroom("mistral", "mistral-tiny", est_prompt_tokens=80, est_completion_tokens=50)
    assert hr2.can_proceed is False

