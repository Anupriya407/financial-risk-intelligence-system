"""
Preprocessing report generator for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class PreprocessingReport:
    """Summary of preprocessing operations."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_removed: int
    missing_values_before: int
    missing_values_after: int
    memory_usage_before: int
    memory_usage_after: int


class ReportGenerator:
    """Generate preprocessing reports."""

    def generate(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> PreprocessingReport:
        """
        Generate a preprocessing report.

        Parameters
        ----------
        before:
            Original dataframe.

        after:
            Processed dataframe.

        Returns
        -------
        PreprocessingReport
        """

        return PreprocessingReport(
            rows_before=len(before),
            rows_after=len(after),
            columns_before=before.shape[1],
            columns_after=after.shape[1],
            duplicates_removed=int(
                before.duplicated().sum()
            ),
            missing_values_before=int(
                before.isna().sum().sum()
            ),
            missing_values_after=int(
                after.isna().sum().sum()
            ),
            memory_usage_before=int(
                before.memory_usage(
                    deep=True,
                ).sum()
            ),
            memory_usage_after=int(
                after.memory_usage(
                    deep=True,
                ).sum()
            ),
        )


report_generator = ReportGenerator()