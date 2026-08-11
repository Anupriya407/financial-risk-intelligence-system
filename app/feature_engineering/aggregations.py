"""
Aggregation utilities for the Financial Risk Intelligence System
feature engineering pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


class FeatureAggregator:
    """Create reusable aggregate features from tabular data."""

    def aggregate(
        self,
        dataframe: pd.DataFrame,
        group_by: str | Sequence[str],
        aggregations: dict[str, list[str]],
        prefix: str,
    ) -> pd.DataFrame:
        """
        Aggregate dataset columns by one or more keys.

        Parameters
        ----------
        dataframe:
            Input DataFrame.

        group_by:
            Column or columns used to group records.

        aggregations:
            Mapping of column names to aggregation functions.

            Example
            -------
            {
                "AMT_CREDIT": ["mean", "max", "min"],
                "DAYS_CREDIT": ["mean", "min", "max"],
            }

        prefix:
            Prefix applied to generated feature names.

        Returns
        -------
        pandas.DataFrame
            Aggregated feature DataFrame.
        """

        if isinstance(group_by, str):
            group_columns = [group_by]
        else:
            group_columns = list(group_by)

        missing_group_columns = [
            column
            for column in group_columns
            if column not in dataframe.columns
        ]

        if missing_group_columns:
            raise ValueError(
                "Missing group-by columns: "
                f"{missing_group_columns}"
            )

        missing_aggregation_columns = [
            column
            for column in aggregations
            if column not in dataframe.columns
        ]

        if missing_aggregation_columns:
            raise ValueError(
                "Missing aggregation columns: "
                f"{missing_aggregation_columns}"
            )

        grouped = dataframe.groupby(
            group_columns,
            dropna=False,
        ).agg(
            aggregations,
        )

        grouped.columns = [
            self._build_feature_name(
                prefix=prefix,
                column=column,
                aggregation=aggregation,
            )
            for column, aggregation in grouped.columns
        ]

        return grouped.reset_index()

    @staticmethod
    def _build_feature_name(
        prefix: str,
        column: str,
        aggregation: str,
    ) -> str:
        """Build a standardized feature name."""

        return (
            f"{prefix}_{column}_{aggregation}"
        )

    def count(
        self,
        dataframe: pd.DataFrame,
        group_by: str | Sequence[str],
        prefix: str,
        feature_name: str = "COUNT",
    ) -> pd.DataFrame:
        """
        Count records for each group.

        Parameters
        ----------
        dataframe:
            Input DataFrame.

        group_by:
            Column or columns used for grouping.

        prefix:
            Feature prefix.

        feature_name:
            Name of the count feature.

        Returns
        -------
        pandas.DataFrame
            Group-level count features.
        """

        if isinstance(group_by, str):
            group_columns = [group_by]
        else:
            group_columns = list(group_by)

        missing_columns = [
            column
            for column in group_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing group-by columns: "
                f"{missing_columns}"
            )

        result = (
            dataframe.groupby(
                group_columns,
                dropna=False,
            )
            .size()
            .reset_index(
                name=f"{prefix}_{feature_name}",
            )
        )

        return result

    def nunique(
        self,
        dataframe: pd.DataFrame,
        group_by: str | Sequence[str],
        column: str,
        prefix: str,
        feature_name: str = "NUNIQUE",
    ) -> pd.DataFrame:
        """
        Count unique values within each group.

        Parameters
        ----------
        dataframe:
            Input DataFrame.

        group_by:
            Column or columns used for grouping.

        column:
            Column for unique-value counting.

        prefix:
            Feature prefix.

        feature_name:
            Name of the resulting feature.

        Returns
        -------
        pandas.DataFrame
            Group-level unique-count features.
        """

        if isinstance(group_by, str):
            group_columns = [group_by]
        else:
            group_columns = list(group_by)

        required_columns = [
            *group_columns,
            column,
        ]

        missing_columns = [
            item
            for item in required_columns
            if item not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing columns: "
                f"{missing_columns}"
            )

        result = (
            dataframe.groupby(
                group_columns,
                dropna=False,
            )[column]
            .nunique(dropna=True)
            .reset_index(
                name=f"{prefix}_{feature_name}",
            )
        )

        return result


feature_aggregator = FeatureAggregator()