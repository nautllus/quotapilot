from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UsageLog(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    provider: str
    model: str
    request_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error_code: Optional[int] = None


class ProviderLimits(BaseModel):
    provider: str
    model: str
    rpm: Optional[int] = None  # requests per minute
    tpm: Optional[int] = None  # tokens per minute
    rpd: Optional[int] = None  # requests per day (24h)
    tpd: Optional[int] = None  # tokens per day (24h)

