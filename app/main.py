"""Main entry point for the Financial Risk Intelligence System (FRIS) FastAPI application."""

from fastapi import FastAPI

from app.api import health_router
from app.config import settings
from app.core import FRISException, fris_exception_handler, lifespan

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan.lifespan,
)

app.add_exception_handler(FRISException, fris_exception_handler)

app.include_router(health_router)
