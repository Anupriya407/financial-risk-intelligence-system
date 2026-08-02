"""
Dataset metadata models for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class DatasetMetadata:
    """Metadata describing a dataset."""

    name: str
    filename: str
    description: str
    primary_key: str | tuple[str, ...]
    path: Path

    @property
    def exists(self) -> bool:
        """Return True if the dataset file exists."""
        return self.path.exists()