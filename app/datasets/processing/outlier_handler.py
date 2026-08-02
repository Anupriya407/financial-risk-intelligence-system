"""
Outlier handler for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from app.datasets.processing.base import BasePreprocessor


OutlierStrategy = Literal[
    "none",
    "clip",
    "remove",
]


class OutlierHandler(BasePreprocessor):
    """Handle numeric outliers using the IQR method."""

    def __init__(
        self,
        strategy: OutlierStrategy = "none",
    ) -> None:
        """Initialize the outlier handler."""

        self.strategy = strategy

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process numeric outliers."""

        if self.strategy == "none":
            return dataframe.copy()

        dataframe = dataframe.copy()

        numeric_columns = dataframe.select_dtypes(
            include="number",
        ).columns

        for column in numeric_columns:
            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if self.strategy == "clip":
                dataframe[column] = dataframe[column].clip(
                    lower=lower,
                    upper=upper,
                )

            elif self.strategy == "remove":
                dataframe = dataframe[
                    dataframe[column].between(
                        lower,
                        upper,
                    )
                ]

        return dataframe.reset_index(
            drop=True,
        )


outlier_handler = OutlierHandler()