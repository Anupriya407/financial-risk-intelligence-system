"""
Dataset quality analysis for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.loader import dataset_loader


class DatasetQualityAnalyzer:
    """Analyze dataset quality."""

    def get_shape(
        self,
        dataset_name: str,
    ) -> tuple[int, int]:
        """Return dataset shape."""

        dataframe = dataset_loader.load(dataset_name)

        return dataframe.shape

    def get_columns(
        self,
        dataset_name: str,
    ) -> list[str]:
        """Return dataset columns."""

        dataframe = dataset_loader.load(
            dataset_name,
            nrows=1,
        )

        return dataframe.columns.tolist()

    def get_dtypes(
        self,
        dataset_name: str,
    ) -> pd.Series:
        """Return dataset data types."""

        dataframe = dataset_loader.load(
            dataset_name,
            nrows=1,
        )

        return dataframe.dtypes

    def get_missing_values(
        self,
        dataset_name: str,
    ) -> pd.Series:
        """Return missing values per column."""

        dataframe = dataset_loader.load(dataset_name)

        return dataframe.isna().sum()

    def get_duplicate_rows(
        self,
        dataset_name: str,
    ) -> int:
        """Return duplicate row count."""

        dataframe = dataset_loader.load(dataset_name)

        return int(dataframe.duplicated().sum())

    def get_memory_usage(
        self,
        dataset_name: str,
    ) -> int:
        """Return dataset memory usage in bytes."""

        dataframe = dataset_loader.load(dataset_name)

        return int(
            dataframe.memory_usage(
                deep=True,
            ).sum(),
        )

    def get_summary(
        self,
        dataset_name: str,
    ) -> dict[str, object]:
        """Return a dataset quality summary."""

        rows, columns = self.get_shape(
            dataset_name,
        )

        return {
            "rows": rows,
            "columns": columns,
            "missing_values": (
                self.get_missing_values(
                    dataset_name,
                ).sum()
            ),
            "duplicate_rows": (
                self.get_duplicate_rows(
                    dataset_name,
                )
            ),
            "memory_usage_bytes": (
                self.get_memory_usage(
                    dataset_name,
                )
            ),
        }


dataset_quality_analyzer = DatasetQualityAnalyzer()