"""
Dataset validation utilities for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.loader import dataset_loader
from app.datasets.registry import dataset_registry


class DatasetValidator:
    """Validate registered datasets."""

    def validate_exists(
        self,
        dataset_name: str,
    ) -> None:
        """Validate that the dataset exists."""

        if not dataset_registry.dataset_exists(dataset_name):
            dataset = dataset_registry.get_dataset(dataset_name)

            raise FileNotFoundError(
                f"Dataset '{dataset.filename}' "
                f"does not exist."
            )

    def validate_not_empty(
        self,
        dataset_name: str,
    ) -> None:
        """Validate that the dataset is not empty."""

        dataframe = dataset_loader.load(
            dataset_name,
            nrows=1,
        )

        if dataframe.empty:
            raise ValueError(
                f"Dataset '{dataset_name}' is empty."
            )

    def validate_columns(
        self,
        dataset_name: str,
        required_columns: list[str],
    ) -> None:
        """Validate required columns."""

        dataframe = dataset_loader.load(
            dataset_name,
            nrows=1,
        )

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

    def validate_primary_key(
        self,
        dataset_name: str,
    ) -> None:
        """Validate that primary key columns exist."""

        dataset = dataset_registry.get_dataset(
            dataset_name,
        )

        dataframe = dataset_loader.load(
            dataset_name,
            nrows=1,
        )

        primary_key = dataset.primary_key

        if isinstance(primary_key, str):
            primary_key = (primary_key,)

        missing = [
            column
            for column in primary_key
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Primary key columns missing: {missing}"
            )

    def validate_dataset(
        self,
        dataset_name: str,
    ) -> None:
        """Run all basic dataset validations."""

        self.validate_exists(dataset_name)
        self.validate_not_empty(dataset_name)
        self.validate_primary_key(dataset_name)


dataset_validator = DatasetValidator()