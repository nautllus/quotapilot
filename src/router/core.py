import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from gateway.schemas import ChatRequest, ChatResponse
from providers.base import ProviderAdapter
from .registry import ProviderRegistry
from .retry import is_retryable, calculate_backoff
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
        # Build candidates list; include budget checks later per-provider
        candidates = await self._list_capable_candidates(request)
        if not candidates:
            logger.error("No capable provider candidates for request")
            raise NoCapableProviderError("No capable provider candidates")

        # Unique providers (preserve first model per provider)
        seen = set()
        queue: List[SelectedProvider] = []
        for prov, model in candidates:
            if prov.name in seen:
                continue
            seen.add(prov.name)
            queue.append(SelectedProvider(provider=prov, model=model))
            if len(queue) >= 3:  # max 3 providers total
                break

        last_error: Optional[Exception] = None
        for sp in queue:
            # Prepare request for this provider
            req = request.model and request or request.copy()
            req.model = sp.model
            # Ensure provider call is non-streaming; gateway handles SSE
            req.stream = False

            # Retry up to 2 attempts on this provider
            attempt = 0
            while attempt < 2:
                attempt += 1
                try:
                    logger.info("Provider %s attempt %d", sp.provider.name, attempt)
                    resp = await sp.provider.chat(req)

                    # Record usage on success
                    if self._budget is not None:
                        try:
                            await self._budget.record_usage(
                                provider=sp.provider.name,
                                model=sp.model,
                                request_tokens=resp.usage.prompt_tokens,
                                response_tokens=resp.usage.completion_tokens,
                                success=True,
                            )
                        except Exception as e:  # pragma: no cover
                            logger.warning("Failed to record usage: %s", e)
                    return resp
                except Exception as e:
                    action, status_code, retry_after = is_retryable(e)
                    last_error = e
                    # Record failed attempt (no tokens)
                    if self._budget is not None:
                        try:
                            await self._budget.record_usage(
                                provider=sp.provider.name,
                                model=sp.model,
                                request_tokens=0,
                                response_tokens=0,
                                success=False,
                                error_code=int(status_code or 0) if status_code is not None else None,
                            )
                        except Exception as rec_err:  # pragma: no cover
                            logger.warning("Failed to record failed usage: %s", rec_err)

                    if action == "retry_same" and attempt < 2:
                        delay = calculate_backoff(attempt, retry_after)
                        logger.info(
                            "Retrying provider %s after %.2fs due to status %s",
                            sp.provider.name,
                            delay,
                            status_code,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if action == "no_retry":
                        logger.error("Non-retryable error from %s: %s", sp.provider.name, e)
                        raise

                    # switch_provider or attempts exhausted
                    logger.info(
                        "Switching provider after error from %s (status %s, attempt %d)",
                        sp.provider.name,
                        status_code,
                        attempt,
                    )
                    break  # move to next provider

        # Exhausted all providers
        logger.error("All providers exhausted; last error: %s", last_error)
        raise NoCapableProviderError("All providers exhausted by retry/failover")

    async def _list_capable_candidates(self, request: ChatRequest) -> List[Tuple[ProviderAdapter, str]]:
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
            logger.error(
                "No capable provider found for request requirements: json=%s tools=%s stream=%s",
                requires_json,
                requires_tools,
                requires_stream,
            )
            return []

        return candidates

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

