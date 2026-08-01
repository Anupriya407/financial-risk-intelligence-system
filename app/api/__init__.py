"""Financial Risk Intelligence System application package."""

from app.api.health import router as health_router
from app.api.v1.api import api_router

__all__ = [
    "api_router",
    "health_router",
]
