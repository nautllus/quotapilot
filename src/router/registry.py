import logging
import os
from typing import Dict, List, Optional

from providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for provider adapters.

    - Holds instantiated providers keyed by provider.name
    - On init, auto-registers MistralAdapter if MISTRAL_API_KEY is set
    """

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderAdapter] = {}
        self._auto_register()

    def register_provider(self, provider: ProviderAdapter) -> None:
        try:
            name = provider.name
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Failed to read provider name: %s", e)
            return
        self._providers[name] = provider
        logger.info("Registered provider: %s", name)

    def get_providers(self) -> List[ProviderAdapter]:
        return list(self._providers.values())

    def get_provider(self, name: str) -> Optional[ProviderAdapter]:
        return self._providers.get(name)

    def _auto_register(self) -> None:
        # Only register Mistral if API key is present
        if os.getenv("MISTRAL_API_KEY"):
            try:
                from providers.mistral import MistralAdapter  # local import to avoid hard dep

                self.register_provider(MistralAdapter())
            except Exception as e:
                logger.warning("Could not auto-register MistralAdapter: %s", e)

