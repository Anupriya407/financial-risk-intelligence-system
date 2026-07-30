"""Health and application information endpoints."""

from fastapi import APIRouter

from app.config import settings

router = APIRouter()


@router.get("/")
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


@router.get("/health")
def health_check() -> dict[str, str]:
    """Perform a simple health check to indicate service availability.

    Returns:
        dict[str, str]: Service status indicating that the application is running.
    """
    return {"status": "healthy"}
