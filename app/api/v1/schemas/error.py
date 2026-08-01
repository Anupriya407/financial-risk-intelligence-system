"""Standard API error response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed information about an API error."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard API error response."""

    success: bool = False
    error: ErrorDetail
    timestamp: datetime
    path: str