"""Global exception handlers for the Financial Risk Intelligence System."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1.schemas.error import ErrorDetail, ErrorResponse
from app.core.logger import logger
from app.exceptions import FRISException


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(FRISException)
    async def fris_exception_handler(
        request: Request,
        exc: FRISException,
    ) -> JSONResponse:
        """Handle application-specific exceptions."""

        logger.warning(
            "Application exception [%s]: %s",
            exc.error_code,
            exc.message,
        )

        response = ErrorResponse(
            error=ErrorDetail(
                code=exc.error_code,
                message=exc.message,
                details=exc.details,
            ),
            timestamp=datetime.now(UTC),
            path=str(request.url.path),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=response.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle request validation errors."""

        logger.warning(
            "Request validation failed: %s",
            exc.errors(),
        )

        response = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed.",
                details={"errors": exc.errors()},
            ),
            timestamp=datetime.now(UTC),
            path=str(request.url.path),
        )

        return JSONResponse(
            status_code=422,
            content=response.model_dump(mode="json"),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""

        logger.warning(
            "HTTP %s: %s",
            exc.status_code,
            exc.detail,
        )

        response = ErrorResponse(
            error=ErrorDetail(
                code="HTTP_ERROR",
                message=str(exc.detail),
                details={},
            ),
            timestamp=datetime.now(UTC),
            path=str(request.url.path),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=response.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected application errors."""

        logger.exception("Unhandled application exception.")

        response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred.",
                details={},
            ),
            timestamp=datetime.now(UTC),
            path=str(request.url.path),
        )

        return JSONResponse(
            status_code=500,
            content=response.model_dump(mode="json"),
        )