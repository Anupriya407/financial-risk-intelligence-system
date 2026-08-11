"""
Base interfaces for the Financial Risk Intelligence System
feature engineering pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeatureEngineer(ABC):
    """Base class for dataset-specific feature engineering."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the feature engineer name."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate features from a dataset.

        Parameters
        ----------
        dataframe:
            Input dataset.

        Returns
        -------
        pandas.DataFrame
            Dataset containing generated features.
        """
        raise NotImplementedError