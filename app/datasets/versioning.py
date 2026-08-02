"""
Dataset versioning for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.datasets.registry import dataset_registry


@dataclass(slots=True, frozen=True)
class DatasetVersion:
    """Represents a dataset version."""

    dataset_name: str
    version: str
    created_at: datetime
    description: str


class DatasetVersionManager:
    """Manage dataset versions."""

    DEFAULT_VERSION = "1.0.0"

    def get_version(
        self,
        dataset_name: str,
    ) -> DatasetVersion:
        """Return version information for a dataset."""

        dataset = dataset_registry.get_dataset(
            dataset_name,
        )

        return DatasetVersion(
            dataset_name=dataset.name,
            version=self.DEFAULT_VERSION,
            created_at=datetime.now(UTC),
            description=dataset.description,
        )

    def list_versions(
        self,
    ) -> list[DatasetVersion]:
        """Return versions for all registered datasets."""

        return [
            self.get_version(dataset.name)
            for dataset in dataset_registry.list_datasets()
        ]


dataset_version_manager = DatasetVersionManager()