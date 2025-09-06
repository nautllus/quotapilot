import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from gateway.schemas import ChatRequest, ChatResponse, ChatChoice, ChatMessage, ChatUsage
from .base import ProviderAdapter

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


class MistralAdapter(ProviderAdapter):
    """Mistral provider adapter implementing ProviderAdapter.

    Focuses on FREE-tier usage by honoring an allowlist from env:
    - MISTRAL_FREE_MODELS: comma-separated list of model IDs considered free-tier

    Authentication:
    - Requires MISTRAL_API_KEY environment variable
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.mistral.ai/v1",
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY", "").strip()
        if not self._api_key:
            logger.warning("MISTRAL_API_KEY is not set; calls will fail until configured.")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client  # may be injected for tests
        self._last_headers: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "mistral"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def models(self) -> List[Dict[str, Any]]:
        """Return free-tier Mistral models and basic capabilities.

        Strategy:
        - Fetch /v1/models
        - Filter by env allowlist MISTRAL_FREE_MODELS if set; otherwise return
          a conservative subset that is commonly available for experimentation
          (best-effort fallback) and present in the /models list.
        - Capabilities returned (best-effort; subject to change):
            supports_json, supports_tools, supports_stream; ctx unknown if not provided
        """
        free_allow = [m.strip() for m in os.getenv("MISTRAL_FREE_MODELS", "").split(",") if m.strip()]
        client = self._get_client()
        try:
            resp = await client.get("/models", headers=self._headers())
            self._last_headers = dict(resp.headers)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else []
        except httpx.HTTPStatusError as e:
            logger.error("Failed to list Mistral models: %s", e)
            # On failure, return an empty list to fail-safe
            return []
        except Exception as e:
            logger.exception("Unexpected error listing Mistral models: %s", e)
            return []

        # Extract available model IDs from API
        available_ids: List[str] = []
        for it in items:
            # OpenAI-style schema typically uses 'id'
            mid = it.get("id") or it.get("name")
            if isinstance(mid, str):
                available_ids.append(mid)

        # If allowlist is set, filter to it (and must exist in available list)
        if free_allow:
            selected = [m for m in available_ids if m in free_allow]
        else:
            # Best-effort defaults seen in Mistral docs/blogs for evaluation
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
                    "ctx": None,  # unknown via public endpoint; could be augmented by a static map
                    "supports_json": True,  # response_format {type: json_object}
                    "supports_tools": True,  # tools parameter supported in docs
                    "supports_stream": True,  # /v1/chat/completions with stream=true
                }
            )
        return results

    async def state(self) -> Dict[str, Any]:
        """Query a lightweight endpoint to infer rate-limit headroom and health.

        We use GET /v1/models to collect rate-limit headers when available.
        """
        client = self._get_client()
        status = "unknown"
        headroom: Dict[str, Union[int, str, None]] = {
            "requests_remaining": None,
            "requests_limit": None,
            "tokens_remaining": None,
            "tokens_limit": None,
            "reset": None,
        }
        try:
            resp = await client.get("/models", headers=self._headers())
            self._last_headers = dict(resp.headers)
            status = "ok" if resp.status_code == 200 else "degraded"
            # Parse common ratelimit headers if present
            h = resp.headers
            headroom["requests_remaining"] = _to_int(h.get("x-ratelimit-remaining-requests") or h.get("ratelimit-remaining"))
            headroom["requests_limit"] = _to_int(h.get("x-ratelimit-limit-requests") or h.get("ratelimit-limit"))
            headroom["tokens_remaining"] = _to_int(h.get("x-ratelimit-remaining-tokens"))
            headroom["tokens_limit"] = _to_int(h.get("x-ratelimit-limit-tokens"))
            headroom["reset"] = h.get("x-ratelimit-reset-requests") or h.get("ratelimit-reset")
        except Exception as e:
            logger.warning("Failed to fetch Mistral state: %s", e)
            status = "unknown"
        return {"status": status, "ratelimit": headroom}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a normalized chat completion with Mistral API.

        - Maps ChatRequest to Mistral's /v1/chat/completions payload
        - Handles JSON mode via response_format
        - Surfaces 429s as MistralRateLimitError with headers
        """
        client = self._get_client()

        # Build messages
        messages: List[Dict[str, Any]] = []
        for m in request.messages:
            msg: Dict[str, Any] = {"role": m.role}
            if m.name is not None:
                msg["name"] = m.name
            # content can be None for tool-call responses; Mistral supports that
            if m.content is not None:
                msg["content"] = m.content
            else:
                msg["content"] = None
            if m.tool_calls is not None:
                # Pass through as-is
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": tc.function,
                    }
                    for tc in m.tool_calls
                ]
            messages.append(msg)

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": bool(request.stream),
        }

        # Optional OpenAI-compatible fields if present
        if request.json:
            payload["response_format"] = {"type": "json_object"}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None and not request.json:
            payload["response_format"] = request.response_format

        try:
            resp = await client.post("/chat/completions", headers=self._headers(), content=json.dumps(payload))
            self._last_headers = dict(resp.headers)
            if resp.status_code == 429:
                raise MistralRateLimitError("Rate limited by Mistral API", 429, dict(resp.headers))
            resp.raise_for_status()
        except MistralRateLimitError:
            raise
        except httpx.HTTPStatusError as e:
            # Surface other HTTP errors with some context
            body = e.response.text if e.response is not None else str(e)
            logger.error("Mistral API error (%s): %s", e.response.status_code if e.response else "?", body)
            raise
        except Exception as e:
            logger.exception("Unexpected error calling Mistral chat: %s", e)
            raise

        data = resp.json()
        # Map to our ChatResponse model
        choices: List[ChatChoice] = []
        for i, ch in enumerate(data.get("choices", [])):
            msg = ch.get("message", {})
            cm = ChatMessage(
                role=msg.get("role", "assistant"),
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),  # Pydantic will accept List[dict] for now
                name=msg.get("name"),
            )
            choices.append(
                ChatChoice(
                    index=ch.get("index", i),
                    message=cm,
                    finish_reason=ch.get("finish_reason"),
                )
            )

        usage = data.get("usage") or {}
        usage_model = ChatUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )

        return ChatResponse(
            id=data.get("id", ""),
            created=int(data.get("created", 0) or 0),
            model=data.get("model", request.model),
            choices=choices,
            usage=usage_model,
        )


def _to_int(value: Optional[Union[str, int]]) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None
