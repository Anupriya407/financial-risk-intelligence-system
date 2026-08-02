"""
Base preprocessing component for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BasePreprocessor(ABC):
    """Base class for all preprocessing components."""

    @abstractmethod
    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process a dataframe.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        Returns
        -------
        pandas.DataFrame
            Processed dataframe.
        """