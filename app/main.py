"""Main entry point for the Financial Risk Intelligence System (FRIS) FastAPI application."""

from fastapi import FastAPI

app = FastAPI(
    title="Financial Risk Intelligence System",
    version="1.0.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Retrieve basic application information and status.

    Returns:
        dict[str, str]: Basic application metadata including name, version, and status.
    """
    return {
        "name": app.title,
        "version": app.version,
        "status": "ok",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Perform a simple health check to indicate service availability.

    Returns:
        dict[str, str]: Service status indicating that the application is running.
    """
    return {"status": "healthy"}

