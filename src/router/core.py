import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from gateway.schemas import ChatRequest, ChatResponse
from providers.base import ProviderAdapter
from .registry import ProviderRegistry
from state.budget import BudgetManager, estimate_tokens_from_request_text

logger = logging.getLogger(__name__)


class NoCapableProviderError(Exception):
    pass


@dataclass
class SelectedProvider:
    provider: ProviderAdapter
    model: str


class Router:
    def __init__(self, registry: ProviderRegistry, budget: Optional[BudgetManager] = None) -> None:
        self._registry = registry
        self._budget = budget

    def set_budget_manager(self, budget: BudgetManager) -> None:
        self._budget = budget

    async def route_request(self, request: ChatRequest) -> ChatResponse:
        selected = await self._select_provider_and_model(request)
        # For initial/basic streaming support, call provider without streaming;
        # the gateway can synthesize SSE from the final response.
        req = request.model and request or request.copy()
        # Ensure the chosen model is set and streaming disabled for provider call
        req.model = selected.model
        req.stream = False
        logger.info("Routing request to provider=%s model=%s", selected.provider.name, selected.model)
        resp = await selected.provider.chat(req)

        # Record usage if budget manager is available
        if self._budget is not None:
            try:
                await self._budget.record_usage(
                    provider=selected.provider.name,
                    model=selected.model,
                    request_tokens=resp.usage.prompt_tokens,
                    response_tokens=resp.usage.completion_tokens,
                    success=True,
                )
            except Exception as e:  # pragma: no cover - do not break on metering failure
                logger.warning("Failed to record usage: %s", e)

        return resp

    async def _select_provider_and_model(self, request: ChatRequest) -> SelectedProvider:
        requires_json = bool(request.json) or (
            isinstance(request.response_format, dict)
            and request.response_format.get("type") == "json_object"
        )
        requires_tools = request.tools is not None
        requires_stream = bool(request.stream)

        provider_hint, desired_model = self._parse_model_hint(request.model)

        candidates: List[Tuple[ProviderAdapter, str]] = []
        for provider in self._registry.get_providers():
            if provider_hint and provider.name != provider_hint:
                continue
            try:
                models = await provider.models()
            except Exception as e:
                logger.warning("Skipping provider %s due to models() error: %s", provider.name, e)
                continue

            for m in models:
                name = m.get("name")
                if not name:
                    continue
                if desired_model and name != desired_model:
                    continue
                if requires_json and not m.get("supports_json", False):
                    continue
                if requires_tools and not m.get("supports_tools", False):
                    continue
                if requires_stream and not m.get("supports_stream", False):
                    continue

                # Check budget headroom if budget manager is configured
                if self._budget is not None:
                    est_prompt = self._estimate_prompt_tokens(request)
                    est_completion = self._estimate_completion_tokens(request)
                    headroom = await self._budget.check_headroom(provider.name, name, est_prompt, est_completion)
                    if not headroom.can_proceed:
                        logger.info("Skipping provider %s model %s due to budget headroom", provider.name, name)
                        continue

                candidates.append((provider, name))

        if not candidates:
            logger.error("No capable provider found for request requirements: json=%s tools=%s stream=%s",
                         requires_json, requires_tools, requires_stream)
            raise NoCapableProviderError("No capable provider available for requested capabilities")

        # For now, return the first candidate.
        provider, model = candidates[0]
        return SelectedProvider(provider=provider, model=model)

    def _estimate_prompt_tokens(self, request: ChatRequest) -> int:
        try:
            text = "\n".join([m.content or "" for m in request.messages])
            return estimate_tokens_from_request_text(text)
        except Exception:
            return 0

    def _estimate_completion_tokens(self, request: ChatRequest) -> int:
        # Fallback to a small default if max_tokens not provided
        return int(request.max_tokens or 256)

    @staticmethod
    def _parse_model_hint(model_field: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse model field which may be:
        - "auto" -> (None, None)
        - "provider:model" -> (provider, model)
        - "model-name" -> (None, model-name)
        """
        if not model_field or model_field == "auto":
            return None, None
        if ":" in model_field:
            parts = model_field.split(":", 1)
            return parts[0], parts[1]
        return None, model_field

