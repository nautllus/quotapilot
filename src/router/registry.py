import logging
import os
from typing import Any, Dict, List, Optional

import yaml

from providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for provider adapters.

    - Holds instantiated providers keyed by provider.name
    - On init, auto-registers providers if API keys are set
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
        models_cfg = self._load_provider_models()

        # Mistral
        if os.getenv("MISTRAL_API_KEY"):
            try:
                from providers.mistral import MistralAdapter  # local import
                self.register_provider(MistralAdapter())
            except Exception as e:
                logger.warning("Could not auto-register MistralAdapter: %s", e)

        # Cerebras
        if os.getenv("CEREBRAS_API_KEY"):
            try:
                from providers.cerebras import CerebrasAdapter  # local import
                cerebras_models = models_cfg.get("cerebras", {}).get("models", [])
                self.register_provider(CerebrasAdapter(models_config=cerebras_models))
            except Exception as e:
                logger.warning("Could not auto-register CerebrasAdapter: %s", e)

    def _load_provider_models(self) -> Dict[str, Any]:
        path = os.getenv(
            "PROVIDER_MODELS_PATH",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "provider_models.yaml"),
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.info("Provider models config not found at %s; proceeding without models config", path)
            return {}
        except Exception as e:
            logger.warning("Failed to load provider models config: %s", e)
            return {}
