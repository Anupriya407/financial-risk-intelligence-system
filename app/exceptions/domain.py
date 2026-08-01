"""Domain-specific exceptions for the Financial Risk Intelligence System."""

from __future__ import annotations

from app.exceptions.base import FRISException


class ResourceNotFoundException(FRISException):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} with identifier '{identifier}' was not found.",
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
        )


class ResourceAlreadyExistsException(FRISException):
    """Raised when attempting to create a duplicate resource."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} with identifier '{identifier}' already exists.",
            error_code="RESOURCE_ALREADY_EXISTS",
            status_code=409,
        )


class ValidationException(FRISException):
    """Raised when business validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
        )


class BusinessRuleViolationException(FRISException):
    """Raised when a business rule is violated."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="BUSINESS_RULE_VIOLATION",
            status_code=422,
        )