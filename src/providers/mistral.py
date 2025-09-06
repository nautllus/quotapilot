import logging
from typing import Any, Dict, List, Optional

import httpx

from .base_openai import BaseOpenAIAdapter, BaseRateLimitError

logger = logging.getLogger(__name__)


class MistralRateLimitError(Exception):
    def __init__(self, message: str, status_code: int, headers: Dict[str, str]):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers
        # Common fields providers use
        self.retry_after = headers.get("retry-after") or headers.get("Retry-After")

    def __str__(self) -> str:  # helpful for logs
        return f"MistralRateLimitError(status_code={self.status_code}, retry_after={self.retry_after})"


class MistralAdapter(BaseOpenAIAdapter):
    """Mistral provider adapter (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.mistral.ai/v1",
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            provider_name="mistral",
            api_key_env="MISTRAL_API_KEY",
            base_url=base_url,
            client=client,
            timeout=timeout,
            api_key=api_key,
        )

    async def models(self) -> List[Dict[str, Any]]:
        """Return free-tier Mistral models and basic capabilities.

        Strategy:
        - Fetch /v1/models then filter by env allowlist MISTRAL_FREE_MODELS.
        """
        import os
        free_allow = [m.strip() for m in os.getenv("MISTRAL_FREE_MODELS", "").split(",") if m.strip()]
        client = self._get_client()
        try:
            resp = await client.get("/models", headers=self._headers())
            self._last_headers = dict(resp.headers)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.warning("Mistral models() failed: %s", e)
            items = []

        available_ids: List[str] = []
        for it in items:
            mid = it.get("id") or it.get("name")
            if isinstance(mid, str):
                available_ids.append(mid)

        if free_allow:
            selected = [m for m in available_ids if m in free_allow]
        else:
            preferred = [
                "mistral-tiny",
                "mistral-small-latest",
                "open-mixtral-8x7b",
                "open-mistral-7b",
                "ministral-3b-latest",
            ]
            selected = [m for m in preferred if m in available_ids]

        results: List[Dict[str, Any]] = []
        for mid in selected:
            results.append(
                {
                    "name": mid,
                    "ctx": None,
                    "supports_json": True,
                    "supports_tools": True,
                    "supports_stream": True,
                }
            )
        return results

    # state() inherited from BaseOpenAIAdapter

    # chat() inherited from BaseOpenAIAdapter
    def _raise_rate_limit_error(self, resp: httpx.Response) -> None:
        raise MistralRateLimitError("Rate limited by Mistral API", resp.status_code, dict(resp.headers))
