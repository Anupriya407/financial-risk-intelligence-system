"""Main entry point for the Financial Risk Intelligence System (FRIS) FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health_router
from app.config import settings
from app.core import FRISException, fris_exception_handler, logger


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

app.add_exception_handler(FRISException, fris_exception_handler)

app.include_router(health_router)

