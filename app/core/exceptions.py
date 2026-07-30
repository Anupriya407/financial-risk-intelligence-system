"""Application-specific base exception definitions."""


class FRISException(Exception):
    """Base exception class for all Financial Risk Intelligence System errors."""

    def __init__(self, message: str) -> None:
        """Initialize the FRIS base exception.

        Args:
            message: Descriptive error message explaining the failure cause.
        """
        super().__init__(message)
        self.message = message
