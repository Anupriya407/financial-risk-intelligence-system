"""
Main entry point for the Financial Risk Intelligence System (FRIS)
FastAPI application.
"""

from fastapi import FastAPI

from app.api import api_router, health_router
from app.config import settings
from app.core import lifespan
from app.core.exception_handlers import register_exception_handlers

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

# Register global exception handlers
register_exception_handlers(app)

app.include_router(health_router)

app.include_router(
    api_router,
    prefix="/api/v1",
)