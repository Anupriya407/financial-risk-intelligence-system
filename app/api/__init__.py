"""API package containing route definitions."""

from app.api.health import router as health_router

__all__ = ["health_router"]
