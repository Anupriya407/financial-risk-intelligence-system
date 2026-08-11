"""
Feature validation utilities for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from app.feature_engineering.config import TARGET_COLUMN


class FeatureValidator:
    """Validate engineered feature datasets."""

    def validate_key_exists(
        self,
        dataframe: pd.DataFrame,
        key: str | Sequence[str],
    ) -> None:
        """
        Validate that feature merge keys exist.

        Parameters
        ----------
        dataframe:
            Feature DataFrame.

        key:
            Merge key or keys.
        """

        if isinstance(key, str):
            keys = [key]
        else:
            keys = list(key)

        missing = [
            column
            for column in keys
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Missing feature merge keys: "
                f"{missing}"
            )

    def validate_unique_keys(
        self,
        dataframe: pd.DataFrame,
        key: str | Sequence[str],
    ) -> None:
        """
        Validate that feature keys uniquely identify rows.

        Parameters
        ----------
        dataframe:
            Feature DataFrame.

        key:
            Feature merge key or keys.
        """

        if isinstance(key, str):
            keys = [key]
        else:
            keys = list(key)

        self.validate_key_exists(
            dataframe,
            keys,
        )

        duplicate_mask = dataframe.duplicated(
            subset=keys,
            keep=False,
        )

        if duplicate_mask.any():
            duplicate_count = int(
                duplicate_mask.sum(),
            )

            raise ValueError(
                "Feature DataFrame contains "
                f"{duplicate_count} rows with "
                "duplicate merge keys."
            )

    def validate_no_target_leakage(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate that engineered features do not
        contain the target column.
        """

        if TARGET_COLUMN in dataframe.columns:
            raise ValueError(
                f"Target leakage detected: "
                f"'{TARGET_COLUMN}' is present in "
                "the engineered feature DataFrame."
            )

    def validate_no_empty_feature_names(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate that feature column names are not empty."""

        empty_names = [
            column
            for column in dataframe.columns
            if not str(column).strip()
        ]

        if empty_names:
            raise ValueError(
                "Feature DataFrame contains empty "
                "column names."
            )

    def validate(
        self,
        dataframe: pd.DataFrame,
        key: str | Sequence[str],
    ) -> None:
        """
        Run all structural feature validations.

        Parameters
        ----------
        dataframe:
            Engineered feature DataFrame.

        key:
            Feature merge key or keys.
        """

        if dataframe.empty:
            raise ValueError(
                "Engineered feature DataFrame is empty."
            )

        self.validate_key_exists(
            dataframe,
            key,
        )

        self.validate_unique_keys(
            dataframe,
            key,
        )

        self.validate_no_target_leakage(
            dataframe,
        )

        self.validate_no_empty_feature_names(
            dataframe,
        )


feature_validator = FeatureValidator()