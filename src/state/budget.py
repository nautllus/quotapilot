import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import yaml
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from .models import UsageLog

logger = logging.getLogger(__name__)


@dataclass
class HeadroomResult:
    can_proceed: bool
    remaining: Dict[str, Optional[int]]  # keys: rpm, rpd, tpm, tpd


class BudgetManager:
    def __init__(
        self,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection: Optional[AsyncIOMotorCollection] = None,
        limits_path: Optional[str] = None,
        limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        if collection is not None:
            self._col = collection
        elif db is not None:
            self._col = db["usage_logs"]
        else:
            raise ValueError("BudgetManager requires a db or collection")

        self._limits = limits or self._load_limits(limits_path)

    def _load_limits(self, limits_path: Optional[str]) -> Dict[str, Any]:
        if not limits_path:
            limits_path = os.getenv(
                "PROVIDER_LIMITS_PATH",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "provider_limits.yaml"),
            )
        try:
            with open(limits_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Provider limits file not found at %s; proceeding without limits", limits_path)
            return {}
        except Exception as e:
            logger.warning("Failed to load provider limits: %s", e)
            return {}

    async def record_usage(
        self,
        provider: str,
        model: str,
        request_tokens: int,
        response_tokens: int,
        success: bool = True,
        error_code: Optional[int] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        doc = UsageLog(
            ts=ts or datetime.utcnow(),
            provider=provider,
            model=model,
            request_tokens=int(request_tokens or 0),
            response_tokens=int(response_tokens or 0),
            total_tokens=int((request_tokens or 0) + (response_tokens or 0)),
            success=success,
            error_code=error_code,
        ).model_dump()
        try:
            await self._col.insert_one(doc)
        except Exception as e:
            logger.warning("Failed to record usage: %s", e)

    async def get_usage_stats(self, provider: str, model: str) -> Dict[str, Dict[str, int]]:
        now = datetime.utcnow()
        minute_ago = now - timedelta(seconds=60)
        day_ago = now - timedelta(hours=24)

        minute = await self._aggregate_window(provider, model, minute_ago)
        day = await self._aggregate_window(provider, model, day_ago)
        return {"minute": minute, "day": day}

    async def _aggregate_window(self, provider: str, model: str, start: datetime) -> Dict[str, int]:
        pipeline = [
            {"$match": {"provider": provider, "model": model, "ts": {"$gte": start}}},
            {"$group": {"_id": None, "requests": {"$sum": 1}, "tokens": {"$sum": "$total_tokens"}}},
        ]
        try:
            cursor = self._col.aggregate(pipeline, allowDiskUse=False)
            docs = await cursor.to_list(length=1)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("Aggregation failed: %s", e)
            return {"requests": 0, "tokens": 0}
        if docs:
            d = docs[0]
            return {"requests": int(d.get("requests", 0) or 0), "tokens": int(d.get("tokens", 0) or 0)}
        return {"requests": 0, "tokens": 0}

    def _get_limits(self, provider: str, model: str) -> Dict[str, Optional[int]]:
        p = self._limits.get(provider, {}) if isinstance(self._limits, dict) else {}
        m = p.get(model)
        if not m:
            # try provider-level default
            m = p.get("default", {})
        if not isinstance(m, dict):
            return {"rpm": None, "rpd": None, "tpm": None, "tpd": None}
        return {
            "rpm": m.get("rpm"),
            "rpd": m.get("rpd"),
            "tpm": m.get("tpm"),
            "tpd": m.get("tpd"),
        }

    async def check_headroom(
        self,
        provider: str,
        model: str,
        est_prompt_tokens: Optional[int] = None,
        est_completion_tokens: Optional[int] = None,
    ) -> HeadroomResult:
        limits = self._get_limits(provider, model)
        if not any(limits.values()):
            # No limits configured -> always allow
            return HeadroomResult(can_proceed=True, remaining={k: None for k in ("rpm", "rpd", "tpm", "tpd")})

        stats = await self.get_usage_stats(provider, model)
        minute = stats["minute"]
        day = stats["day"]

        est_total = int((est_prompt_tokens or 0) + (est_completion_tokens or 0))

        can = True
        remaining = {"rpm": None, "rpd": None, "tpm": None, "tpd": None}

        rpm = limits.get("rpm")
        if rpm is not None:
            remaining["rpm"] = max(0, rpm - minute["requests"])
            if minute["requests"] >= rpm:
                can = False

        rpd = limits.get("rpd")
        if rpd is not None:
            remaining["rpd"] = max(0, rpd - day["requests"])
            if day["requests"] >= rpd:
                can = False

        tpm = limits.get("tpm")
        if tpm is not None:
            remaining["tpm"] = max(0, tpm - minute["tokens"])
            if minute["tokens"] + est_total > tpm:
                can = False

        tpd = limits.get("tpd")
        if tpd is not None:
            remaining["tpd"] = max(0, tpd - day["tokens"])
            if day["tokens"] + est_total > tpd:
                can = False

        return HeadroomResult(can_proceed=can, remaining=remaining)


def estimate_tokens_from_request_text(text: str) -> int:
    # Very rough heuristic: ~4 characters per token
    return max(1, int(len(text) / 4))

