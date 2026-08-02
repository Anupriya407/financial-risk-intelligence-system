"""
Dataset preprocessing pipeline for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.processing.datatype_handler import DatatypeHandler
from app.datasets.processing.duplicate_handler import DuplicateHandler
from app.datasets.processing.missing_value_handler import (
    MissingValueHandler,
)
from app.datasets.processing.outlier_handler import OutlierHandler
from app.datasets.processing.report_generator import (
    PreprocessingReport,
    ReportGenerator,
)


class PreprocessingPipeline:
    """Run the complete preprocessing pipeline."""

    def __init__(
        self,
        duplicate_handler: DuplicateHandler | None = None,
        missing_value_handler: MissingValueHandler | None = None,
        datatype_handler: DatatypeHandler | None = None,
        outlier_handler: OutlierHandler | None = None,
        report_generator: ReportGenerator | None = None,
    ) -> None:
        """Initialize the preprocessing pipeline."""

        self.duplicate_handler = (
            duplicate_handler or DuplicateHandler()
        )

        self.missing_value_handler = (
            missing_value_handler or MissingValueHandler()
        )

        self.datatype_handler = (
            datatype_handler or DatatypeHandler()
        )

        self.outlier_handler = (
            outlier_handler or OutlierHandler()
        )

        self.report_generator = (
            report_generator or ReportGenerator()
        )

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, PreprocessingReport]:
        """
        Run the preprocessing pipeline.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        Returns
        -------
        tuple[pandas.DataFrame, PreprocessingReport]
            Processed dataframe and preprocessing report.
        """

        original_dataframe = dataframe.copy()

        dataframe = self.duplicate_handler.process(
            dataframe,
        )

        dataframe = self.missing_value_handler.process(
            dataframe,
        )

        dataframe = self.datatype_handler.process(
            dataframe,
        )

        dataframe = self.outlier_handler.process(
            dataframe,
        )

        report = self.report_generator.generate(
            before=original_dataframe,
            after=dataframe,
        )

        return dataframe, report


preprocessing_pipeline = PreprocessingPipeline()