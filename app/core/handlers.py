"""Global exception handlers for the application."""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import FRISException
from app.core.logger import logger


async def fris_exception_handler(request: Request, exc: FRISException) -> JSONResponse:
    """Handle uncaught FRISException instances across the application.

    Args:
        request: The incoming FastAPI HTTP request.
        exc: The raised FRISException instance.

    Returns:
        JSONResponse: Clean HTTP error response with status code 500.
    """
    logger.error("FRISException encountered on path %s: %s", request.url.path, exc.message)
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message},
    )
