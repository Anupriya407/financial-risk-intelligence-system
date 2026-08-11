"""
Feature engineering pipeline for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.validator import feature_validator


class FeatureEngineeringPipeline:
    """Orchestrate dataset-specific feature engineers."""

    def __init__(
        self,
        engineers: Sequence[BaseFeatureEngineer] | None = None,
    ) -> None:
        """
        Initialize the feature engineering pipeline.

        Parameters
        ----------
        engineers:
            Optional collection of feature engineers.
        """

        self._engineers = list(
            engineers or [],
        )

    def register(
        self,
        engineer: BaseFeatureEngineer,
    ) -> None:
        """
        Register a feature engineer.

        Parameters
        ----------
        engineer:
            Feature engineer implementing
            BaseFeatureEngineer.
        """

        if any(
            existing.name == engineer.name
            for existing in self._engineers
        ):
            raise ValueError(
                f"Feature engineer '{engineer.name}' "
                "is already registered."
            )

        self._engineers.append(
            engineer,
        )

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute all registered feature engineers.

        Parameters
        ----------
        dataframe:
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing generated features.
        """

        result = dataframe.copy(
            deep=True,
        )

        for engineer in self._engineers:
            result = engineer.transform(
                result,
            )

        return result

    def transform_features(
        self,
        dataframe: pd.DataFrame,
        key: str | Sequence[str],
    ) -> pd.DataFrame:
        """
        Generate and validate features.

        Parameters
        ----------
        dataframe:
            Input DataFrame.

        key:
            Merge key or keys used for feature validation.

        Returns
        -------
        pandas.DataFrame
            Validated engineered feature DataFrame.
        """

        result = self.transform(
            dataframe,
        )

        feature_validator.validate(
            result,
            key,
        )

        return result

    def list_engineers(self) -> list[str]:
        """Return registered feature engineer names."""

        return [
            engineer.name
            for engineer in self._engineers
        ]


feature_engineering_pipeline = (
    FeatureEngineeringPipeline()
)