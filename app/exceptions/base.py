"""Base exception for the Financial Risk Intelligence System."""

from __future__ import annotations

from typing import Any


class FRISException(Exception):
    """Base exception for all application-specific exceptions."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "APPLICATION_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
