"""
Dataset pipeline for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.loader import dataset_loader
from app.datasets.profiler import dataset_profiler
from app.datasets.quality import dataset_quality_analyzer
from app.datasets.registry import dataset_registry

from app.datasets.validator import dataset_validator
from app.datasets.versioning import (
    dataset_version_manager,
)


class DatasetPipeline:
    """Orchestrates dataset operations."""

    def validate(self) -> None:
        """Validate the complete dataset infrastructure."""

        dataset_registry.validate_registry()

        for dataset_name in dataset_registry.list_dataset_names():
            dataset_validator.validate_dataset(
                dataset_name,
            )

        # Relationship validation is intentionally
        # skipped in FRIS V1.
        #
        # The Home Credit auxiliary datasets contain
        # records beyond application_train.csv, and
        # relationship analysis will be included in
        # the data quality reporting phase instead.

    def load(
        self,
        dataset_name: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Validate and load a dataset.

        Parameters
        ----------
        dataset_name:
            Registered dataset name.

        **kwargs:
            Additional keyword arguments passed to
            pandas.read_csv().

        Returns
        -------
        pandas.DataFrame
        """

        dataset_validator.validate_dataset(
            dataset_name,
        )

        return dataset_loader.load(
            dataset_name,
            **kwargs,
        )

    def profile(
        self,
        dataset_name: str,
    ) -> dict[str, object]:
        """
        Generate a dataset profile.

        Parameters
        ----------
        dataset_name:
            Registered dataset name.

        Returns
        -------
        dict[str, object]
            Dataset profiling information.
        """

        dataset_validator.validate_dataset(
            dataset_name,
        )

        return dataset_profiler.profile(
            dataset_name,
        )

    def quality_report(
        self,
        dataset_name: str,
    ) -> dict[str, object]:
        """
        Return a dataset quality report.

        Parameters
        ----------
        dataset_name:
            Registered dataset name.

        Returns
        -------
        dict[str, object]
            Dataset quality summary.
        """

        dataset_validator.validate_dataset(
            dataset_name,
        )

        return dataset_quality_analyzer.get_summary(
            dataset_name,
        )

    def version(
        self,
        dataset_name: str,
    ):
        """
        Return dataset version information.

        Parameters
        ----------
        dataset_name:
            Registered dataset name.
        """

        return dataset_version_manager.get_version(
            dataset_name,
        )


dataset_pipeline = DatasetPipeline()