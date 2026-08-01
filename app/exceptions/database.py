"""Database-related exceptions for the Financial Risk Intelligence System."""

from __future__ import annotations

from app.exceptions.base import FRISException


class DatabaseException(FRISException):
    """Base exception for persistence-related errors."""

    def __init__(self, message: str = "A database error occurred.") -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
        )


class DatabaseConnectionException(DatabaseException):
    """Raised when the database cannot be reached."""

    def __init__(self) -> None:
        super().__init__(
            "Unable to establish a database connection.",
        )
        self.error_code = "DATABASE_CONNECTION_ERROR"


class DatabaseIntegrityException(DatabaseException):
    """Raised when a database integrity constraint is violated."""

    def __init__(self, message: str = "Database integrity constraint violated.") -> None:
        super().__init__(message)
        self.error_code = "DATABASE_INTEGRITY_ERROR"


class DatabaseTransactionException(DatabaseException):
    """Raised when a database transaction fails."""

    def __init__(self) -> None:
        super().__init__(
            "Database transaction failed.",
        )
        self.error_code = "DATABASE_TRANSACTION_ERROR"