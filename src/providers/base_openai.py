import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from gateway.schemas import ChatRequest, ChatResponse, ChatChoice, ChatMessage, ChatUsage
from providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


class BaseRateLimitError(Exception):
    def __init__(self, message: str, status_code: int, headers: Dict[str, str]):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers
        self.retry_after = headers.get("retry-after") or headers.get("Retry-After")

    def __str__(self) -> str:
        return f"BaseRateLimitError(status_code={self.status_code}, retry_after={self.retry_after})"


class BaseOpenAIAdapter(ProviderAdapter):
    """Base adapter for OpenAI-compatible chat completions providers.

    Subclasses should provide provider name, base_url, and api_key_env.
    They can override models() to list supported models via API or config.
    """

    def __init__(
        self,
        provider_name: str,
        api_key_env: str,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ) -> None:
        self._provider_name = provider_name
        self._api_key_env = api_key_env
        self._api_key = api_key or os.getenv(api_key_env, "").strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._last_headers: Dict[str, str] = {}
        if not self._api_key:
            logger.warning("%s not set; calls will fail until configured.", api_key_env)

    @property
    def name(self) -> str:
        return self._provider_name

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

    async def state(self) -> Dict[str, Any]:
        client = self._get_client()
        status = "unknown"
        headroom: Dict[str, Any] = {
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
            h = resp.headers
            # Support both x-ratelimit-* and ratelimit-*
            headroom["requests_remaining"] = _to_int(
                h.get("x-ratelimit-remaining-requests") or h.get("ratelimit-remaining")
            )
            headroom["requests_limit"] = _to_int(
                h.get("x-ratelimit-limit-requests") or h.get("ratelimit-limit")
            )
            headroom["tokens_remaining"] = _to_int(h.get("x-ratelimit-remaining-tokens"))
            headroom["tokens_limit"] = _to_int(h.get("x-ratelimit-limit-tokens"))
            headroom["reset"] = h.get("x-ratelimit-reset-requests") or h.get("ratelimit-reset")
        except Exception as e:
            logger.warning("Failed to fetch %s state: %s", self._provider_name, e)
        return {"status": status, "ratelimit": headroom}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        client = self._get_client()

        messages: List[Dict[str, Any]] = []
        for m in request.messages:
            msg: Dict[str, Any] = {"role": m.role}
            if m.name is not None:
                msg["name"] = m.name
            msg["content"] = m.content if m.content is not None else None
            if m.tool_calls is not None:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": tc.type, "function": tc.function} for tc in m.tool_calls
                ]
            messages.append(msg)

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": bool(request.stream),
        }
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
                self._raise_rate_limit_error(resp)
            resp.raise_for_status()
        except BaseRateLimitError:
            raise
        except httpx.HTTPStatusError as e:
            logger.error("%s API error (%s): %s", self._provider_name, e.response.status_code if e.response else "?", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error calling %s chat: %s", self._provider_name, e)
            raise

        data = resp.json()
        choices: List[ChatChoice] = []
        for i, ch in enumerate(data.get("choices", [])):
            msg = ch.get("message", {})
            cm = ChatMessage(
                role=msg.get("role", "assistant"),
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),
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

    def _raise_rate_limit_error(self, resp: httpx.Response) -> None:
        raise BaseRateLimitError("Rate limited", resp.status_code, dict(resp.headers))


def _to_int(value: Optional[Any]) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None
