"""Application configuration module for the Financial Risk Intelligence System."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Centralized configuration settings for the application."""

    app_name: str = "Financial Risk Intelligence System"
    version: str = "1.0.0"


settings = Settings()
