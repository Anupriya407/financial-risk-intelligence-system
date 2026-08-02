"""
Base dataset workflow for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path

from app.datasets.pipeline import dataset_pipeline
from app.datasets.processing.pipeline import (
    preprocessing_pipeline,
)
from app.datasets.storage.manager import storage_manager


@dataclass(slots=True)
class WorkflowResult:
    """Result of executing a dataset workflow."""

    dataset_name: str
    rows: int
    columns: int
    parquet_path: Path
    profile_path: Path
    quality_report_path: Path
    preprocessing_report_path: Path


class BaseDatasetWorkflow(ABC):
    """Base workflow for dataset engineering."""

    DATASET_NAME: str = ""

    def run(self) -> WorkflowResult:
        """Execute the complete dataset workflow."""

        if not self.DATASET_NAME:
            raise ValueError(
                "DATASET_NAME must be defined "
                "by subclasses."
            )

        # Validate
        dataset_pipeline.validate()

        # Load
        dataframe = dataset_pipeline.load(
            self.DATASET_NAME,
        )

        # Profile
        profile = dataset_pipeline.profile(
            self.DATASET_NAME,
        )

        # Quality
        quality = dataset_pipeline.quality_report(
            self.DATASET_NAME,
        )

        # Preprocess
        processed_dataframe, preprocessing_report = (
            preprocessing_pipeline.process(
                dataframe,
            )
        )

        # Save dataset
        parquet_path = storage_manager.save_dataset(
            processed_dataframe,
            self.DATASET_NAME,
        )

        # Save reports
        profile_path = storage_manager.save_report(
            profile,
            f"{self.DATASET_NAME}_profile",
        )

        quality_path = storage_manager.save_report(
            quality,
            f"{self.DATASET_NAME}_quality",
        )

        preprocessing_path = storage_manager.save_report(
            asdict(preprocessing_report),
            f"{self.DATASET_NAME}_preprocessing",
        )

        return WorkflowResult(
            dataset_name=self.DATASET_NAME,
            rows=processed_dataframe.shape[0],
            columns=processed_dataframe.shape[1],
            parquet_path=parquet_path,
            profile_path=profile_path,
            quality_report_path=quality_path,
            preprocessing_report_path=preprocessing_path,
        )