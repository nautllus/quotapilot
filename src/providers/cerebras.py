import os
import logging
from typing import Any, Dict, List, Optional

import httpx
import yaml

from .base_openai import BaseOpenAIAdapter

logger = logging.getLogger(__name__)


class CerebrasAdapter(BaseOpenAIAdapter):
    """Cerebras provider adapter (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.cerebras.ai/v1",
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
        models_config: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            provider_name="cerebras",
            api_key_env="CEREBRAS_API_KEY",
            base_url=base_url,
            client=client,
            timeout=timeout,
            api_key=api_key,
        )
        self._models_config = models_config

    async def models(self) -> List[Dict[str, Any]]:
        models_cfg = self._models_config or self._load_models_from_yaml()
        results: List[Dict[str, Any]] = []
        for m in models_cfg:
            name = m.get("name")
            if not name:
                continue
            results.append(
                {
                    "name": name,
                    "ctx": m.get("context_window"),
                    "supports_json": bool(m.get("supports_json", True)),
                    "supports_tools": bool(m.get("supports_tools", False)),
                    "supports_stream": bool(m.get("supports_stream", True)),
                }
            )
        return results

    def _load_models_from_yaml(self) -> List[Dict[str, Any]]:
        path = os.getenv(
            "PROVIDER_MODELS_PATH",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "provider_models.yaml"),
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                cerebras = data.get("cerebras", {})
                return list(cerebras.get("models", []))
        except FileNotFoundError:
            logger.warning("Provider models file not found at %s; returning empty list", path)
            return []
        except Exception as e:
            logger.warning("Failed to load provider models YAML: %s", e)
            return []
