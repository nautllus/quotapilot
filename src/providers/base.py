from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List

try:
    # Optional import for type hints only to avoid runtime dependency loops
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from src.gateway.schemas import ChatRequest, ChatResponse
except Exception:  # pragma: no cover - strictly for safety in skeleton
    pass


class ProviderAdapter(ABC):
    """Abstract base class for provider adapters.

    Implementations should be side-effect free constructors and expose async methods
    compatible with the router. Keep return shapes simple (dicts / Pydantic models).
    """

    name: str = "provider"

    @abstractmethod
    async def models(self) -> List[Dict[str, Any]]:
        """Return a list of supported models with basic capability metadata.

        Example item:
        {
            "name": "provider:model-id",
            "ctx": 128000,
            "supports_json": True,
            "supports_tools": True,
            "supports_stream": True,
        }
        """

    @abstractmethod
    async def state(self) -> Dict[str, Any]:
        """Return provider state such as rolling RPM/RPD headroom and health.

        Example:
        {"rpm_headroom": 60, "rpd_headroom": 5000, "reliability": 0.997}
        """

    @abstractmethod
    async def chat(self, request: "ChatRequest") -> "ChatResponse":
        """Execute a normalized chat completion request and return a normalized response."""

