"""Main entry point for the Financial Risk Intelligence System (FRIS) FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.core import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifespan events.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("Application starting...")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Retrieve basic application information and status.

    Returns:
        dict[str, str]: Basic application metadata including name, version, and status.
    """
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "ok",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Perform a simple health check to indicate service availability.

    Returns:
        dict[str, str]: Service status indicating that the application is running.
    """
    return {"status": "healthy"}
