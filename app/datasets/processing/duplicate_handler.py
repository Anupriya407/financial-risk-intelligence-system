"""
Duplicate row handler for the Financial Risk Intelligence System.
"""

from __future__ import annotations

import pandas as pd

from app.datasets.processing.base import BasePreprocessor


class DuplicateHandler(BasePreprocessor):
    """Handle duplicate rows in datasets."""

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        Returns
        -------
        pandas.DataFrame
            Dataframe with duplicate rows removed.
        """

        return dataframe.drop_duplicates(
            ignore_index=True,
        )


duplicate_handler = DuplicateHandler()