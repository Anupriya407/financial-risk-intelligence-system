"""
Missing value handler for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from app.datasets.processing.base import BasePreprocessor


MissingValueStrategy = Literal[
    "drop_rows",
    "drop_columns",
    "mean",
    "median",
    "mode",
    "constant",
]


class MissingValueHandler(BasePreprocessor):
    """Handle missing values in datasets."""

    def __init__(
        self,
        strategy: MissingValueStrategy = "mean",
        fill_value: Any = None,
    ) -> None:
        """Initialize the missing value handler."""

        self.strategy = strategy
        self.fill_value = fill_value

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Handle missing values according to the selected strategy.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        Returns
        -------
        pandas.DataFrame
            Processed dataframe.
        """

        dataframe = dataframe.copy()

        if self.strategy == "drop_rows":
            return dataframe.dropna()

        if self.strategy == "drop_columns":
            return dataframe.dropna(axis=1)

        if self.strategy == "mean":
            numeric_columns = dataframe.select_dtypes(
                include="number",
            ).columns

            dataframe[numeric_columns] = dataframe[
                numeric_columns
            ].fillna(
                dataframe[numeric_columns].mean(),
            )

            return dataframe

        if self.strategy == "median":
            numeric_columns = dataframe.select_dtypes(
                include="number",
            ).columns

            dataframe[numeric_columns] = dataframe[
                numeric_columns
            ].fillna(
                dataframe[numeric_columns].median(),
            )

            return dataframe

        if self.strategy == "mode":
            for column in dataframe.columns:
                mode = dataframe[column].mode()

                if not mode.empty:
                    dataframe[column] = dataframe[column].fillna(
                        mode.iloc[0],
                    )

            return dataframe

        if self.strategy == "constant":
            return dataframe.fillna(self.fill_value)

        raise ValueError(
            f"Unknown missing value strategy: {self.strategy}",
        )


missing_value_handler = MissingValueHandler()