import math
from typing import Optional, Tuple


class RetryableError(Exception):
    """Base class for retry-inducing errors."""


def _extract_status_code(exc: Exception) -> Optional[int]:
    # Common patterns: custom exc.status_code, httpx.HTTPStatusError.response.status_code
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _extract_retry_after(exc: Exception) -> Optional[str]:
    # Custom attribute
    ra = getattr(exc, "retry_after", None)
    if ra:
        return str(ra)
    # httpx.HTTPStatusError -> response.headers
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            return str(retry_after)
    return None


def is_retryable(exc: Exception) -> Tuple[str, Optional[int], Optional[str]]:
    """Classify an exception for retry handling.

    Returns a tuple (action, status_code, retry_after) where action is one of:
    - "retry_same": retry same provider after backoff
    - "switch_provider": try next provider
    - "no_retry": do not retry
    """
    status_code = _extract_status_code(exc)
    retry_after = _extract_retry_after(exc)

    if status_code == 429:
        return "retry_same", status_code, retry_after
    if status_code in (502, 503, 504):
        return "switch_provider", status_code, retry_after
    if status_code in (400, 401, 403, 404):
        return "no_retry", status_code, retry_after
    # Default: if no status, assume switch provider for safety on 5xx-like
    return "switch_provider", status_code, retry_after


def calculate_backoff(attempt: int, retry_after_header: Optional[str] = None) -> float:
    """Compute backoff seconds for a retry attempt.

    If Retry-After header is provided, prefer it; otherwise exponential backoff 1s, 2s.
    """
    if retry_after_header:
        try:
            return float(int(retry_after_header))
        except Exception:
            # Could parse HTTP-date, but keep simple for now
            pass
    # attempt is 1-based
    return float(min(2, max(1, 2 ** (attempt - 1))))
