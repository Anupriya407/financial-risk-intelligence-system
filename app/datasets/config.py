"""Dataset configuration for the Financial Risk Intelligence System."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

INTERIM_DATA_DIR = DATA_DIR / "interim"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

METADATA_DIR = DATA_DIR / "metadata"