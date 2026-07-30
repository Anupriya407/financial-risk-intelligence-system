"""Application lifecycle management for the Financial Risk Intelligence System."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifespan events.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("Application starting...")
    yield
    logger.info("Application shutting down...")
