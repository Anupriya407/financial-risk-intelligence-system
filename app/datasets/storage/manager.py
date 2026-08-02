"""
Storage manager for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.datasets.config import (
    INTERIM_DATA_DIR,
    METADATA_DIR,
)
from app.datasets.storage.json_writer import JsonWriter
from app.datasets.storage.parquet_writer import ParquetWriter


class StorageManager:
    """Manage dataset storage operations."""

    def __init__(
        self,
        parquet_writer: ParquetWriter | None = None,
        json_writer: JsonWriter | None = None,
    ) -> None:
        """Initialize the storage manager."""

        self.parquet_writer = (
            parquet_writer or ParquetWriter()
        )

        self.json_writer = (
            json_writer or JsonWriter()
        )

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> Path:
        """
        Save a processed dataset as a Parquet file.

        Parameters
        ----------
        dataframe:
            DataFrame to save.

        dataset_name:
            Dataset name.

        Returns
        -------
        pathlib.Path
            Output file path.
        """

        output_path = (
            INTERIM_DATA_DIR /
            f"{dataset_name}.parquet"
        )

        self.parquet_writer.write(
            dataframe,
            output_path,
        )

        return output_path

    def save_report(
        self,
        report: dict[str, Any],
        report_name: str,
    ) -> Path:
        """
        Save a JSON report.

        Parameters
        ----------
        report:
            Report dictionary.

        report_name:
            Report filename without extension.

        Returns
        -------
        pathlib.Path
            Output file path.
        """

        output_path = (
            METADATA_DIR /
            f"{report_name}.json"
        )

        self.json_writer.write(
            report,
            output_path,
        )

        return output_path


storage_manager = StorageManager()