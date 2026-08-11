"""
Feature merging utilities for the Financial Risk Intelligence System.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


class FeatureMerger:
    """Merge engineered feature tables into a base dataset."""

    def merge(
        self,
        base_dataframe: pd.DataFrame,
        feature_dataframe: pd.DataFrame,
        on: str | Sequence[str],
        how: str = "left",
        validate: str = "one_to_one",
    ) -> pd.DataFrame:
        """
        Merge an engineered feature table into a base DataFrame.

        Parameters
        ----------
        base_dataframe:
            Base dataset receiving the features.

        feature_dataframe:
            Engineered feature dataset.

        on:
            Column or columns used as merge keys.

        how:
            Merge strategy. Defaults to left join.

        validate:
            Pandas merge validation rule.

        Returns
        -------
        pandas.DataFrame
            Merged DataFrame.
        """

        if isinstance(on, str):
            merge_columns = [on]
        else:
            merge_columns = list(on)

        missing_base_columns = [
            column
            for column in merge_columns
            if column not in base_dataframe.columns
        ]

        if missing_base_columns:
            raise ValueError(
                "Missing merge columns in base dataset: "
                f"{missing_base_columns}"
            )

        missing_feature_columns = [
            column
            for column in merge_columns
            if column not in feature_dataframe.columns
        ]

        if missing_feature_columns:
            raise ValueError(
                "Missing merge columns in feature dataset: "
                f"{missing_feature_columns}"
            )

        overlapping_columns = (
            set(base_dataframe.columns)
            & set(feature_dataframe.columns)
        )

        overlapping_non_key_columns = (
            overlapping_columns - set(merge_columns)
        )

        if overlapping_non_key_columns:
            raise ValueError(
                "Feature DataFrame contains columns that "
                "already exist in the base DataFrame: "
                f"{sorted(overlapping_non_key_columns)}"
            )

        if feature_dataframe.duplicated(
            subset=merge_columns,
        ).any():
            raise ValueError(
                "Feature DataFrame contains duplicate "
                "merge keys. Feature tables must contain "
                "one row per merge key before merging."
            )

        return base_dataframe.merge(
            feature_dataframe,
            on=merge_columns,
            how=how,
            validate=validate,
        )

    def merge_customer_features(
        self,
        application_dataframe: pd.DataFrame,
        feature_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge customer-level features into applications.

        Parameters
        ----------
        application_dataframe:
            Application-level dataset.

        feature_dataframe:
            Customer-level engineered features.

        Returns
        -------
        pandas.DataFrame
            Application dataset with engineered features.
        """

        return self.merge(
            base_dataframe=application_dataframe,
            feature_dataframe=feature_dataframe,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )


feature_merger = FeatureMerger()