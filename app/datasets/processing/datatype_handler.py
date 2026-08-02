"""
Data type handler for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from app.datasets.processing.base import BasePreprocessor


NumericType = Literal[
    "int64",
    "float64",
]

StringType = Literal[
    "string",
]

BooleanType = Literal[
    "bool",
]

DatetimeType = Literal[
    "datetime64[ns]",
]

SupportedType = (
    NumericType
    | StringType
    | BooleanType
    | DatetimeType
)


class DatatypeHandler(BasePreprocessor):
    """Convert dataframe columns to expected data types."""

    def __init__(
        self,
        schema: dict[str, SupportedType] | None = None,
    ) -> None:
        """Initialize the datatype handler."""

        self.schema = schema or {}

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert dataframe columns according
        to the configured schema.
        """

        dataframe = dataframe.copy()

        for column, dtype in self.schema.items():
            if column not in dataframe.columns:
                continue

            if dtype == "datetime64[ns]":
                dataframe[column] = pd.to_datetime(
                    dataframe[column],
                    errors="coerce",
                )

            else:
                dataframe[column] = dataframe[column].astype(
                    dtype,
                    errors="ignore",
                )

        return dataframe


datatype_handler = DatatypeHandler()