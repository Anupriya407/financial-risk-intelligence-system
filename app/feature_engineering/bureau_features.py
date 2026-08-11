"""
Bureau-level feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.aggregations import feature_aggregator
from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import APPLICATION_KEY


class BureauFeatureEngineer(BaseFeatureEngineer):
    """Generate customer-level features from bureau history."""

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "bureau"

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate customer-level bureau features.

        Parameters
        ----------
        dataframe:
            Bureau dataset.

        Returns
        -------
        pandas.DataFrame
            One row per SK_ID_CURR containing
            engineered bureau features.
        """

        self._validate_required_columns(
            dataframe,
        )

        result = self._create_aggregated_features(
            dataframe,
        )

        result = self._create_account_status_features(
            dataframe,
            result,
        )

        result = self._create_overdue_features(
            dataframe,
            result,
        )

        result = self._create_credit_history_features(
            dataframe,
            result,
        )

        result.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return result

    def _create_aggregated_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create numerical bureau aggregations."""

        aggregations: dict[str, list[str]] = {}

        candidate_columns = {
            "AMT_CREDIT_SUM": [
                "sum",
                "mean",
                "max",
                "min",
            ],
            "AMT_CREDIT_SUM_DEBT": [
                "sum",
                "mean",
                "max",
            ],
            "AMT_CREDIT_SUM_OVERDUE": [
                "sum",
                "mean",
                "max",
            ],
            "AMT_CREDIT_MAX_OVERDUE": [
                "max",
                "mean",
            ],
            "DAYS_CREDIT": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_CREDIT_ENDDATE": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_ENDDATE_FACT": [
                "mean",
                "min",
                "max",
            ],
            "CNT_CREDIT_PROLONG": [
                "sum",
                "mean",
                "max",
            ],
        }

        for column, functions in candidate_columns.items():
            if column in dataframe.columns:
                aggregations[column] = functions

        # A minimal/test dataset may contain only SK_ID_CURR.
        # In that situation there are no numerical columns
        # available for aggregation.
        if not aggregations:
            return (
                dataframe[
                    [APPLICATION_KEY]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        return feature_aggregator.aggregate(
            dataframe=dataframe,
            group_by=APPLICATION_KEY,
            aggregations=aggregations,
            prefix="FE_BUREAU",
        )

    def _create_account_status_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create account status and count features."""

        grouped = dataframe.groupby(
            APPLICATION_KEY,
            dropna=False,
        )

        account_counts = (
            grouped.size()
            .reset_index(
                name="FE_BUREAU_ACCOUNT_COUNT",
            )
        )

        result = result.merge(
            account_counts,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

        if "CREDIT_ACTIVE" not in dataframe.columns:
            return result

        active_counts = (
            dataframe.assign(
                _active=(
                    dataframe["CREDIT_ACTIVE"]
                    == "Active"
                ).astype(int),
            )
            .groupby(
                APPLICATION_KEY,
                dropna=False,
            )["_active"]
            .sum()
            .reset_index(
                name="FE_BUREAU_ACTIVE_COUNT",
            )
        )

        result = result.merge(
            active_counts,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

        closed_counts = (
            dataframe.assign(
                _closed=(
                    dataframe["CREDIT_ACTIVE"]
                    == "Closed"
                ).astype(int),
            )
            .groupby(
                APPLICATION_KEY,
                dropna=False,
            )["_closed"]
            .sum()
            .reset_index(
                name="FE_BUREAU_CLOSED_COUNT",
            )
        )

        result = result.merge(
            closed_counts,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

        result["FE_BUREAU_ACTIVE_RATIO"] = (
            self._safe_divide(
                result["FE_BUREAU_ACTIVE_COUNT"],
                result["FE_BUREAU_ACCOUNT_COUNT"],
            )
        )

        result["FE_BUREAU_CLOSED_RATIO"] = (
            self._safe_divide(
                result["FE_BUREAU_CLOSED_COUNT"],
                result["FE_BUREAU_ACCOUNT_COUNT"],
            )
        )

        return result

    def _create_overdue_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create overdue-related features."""

        if "AMT_CREDIT_SUM_OVERDUE" not in dataframe.columns:
            return result

        overdue = dataframe.assign(
            _overdue=(
                dataframe["AMT_CREDIT_SUM_OVERDUE"]
                > 0
            ).astype(int),
        )

        overdue_features = (
            overdue.groupby(
                APPLICATION_KEY,
                dropna=False,
            )["_overdue"]
            .agg(
                [
                    "sum",
                    "mean",
                ],
            )
            .reset_index()
        )

        overdue_features.rename(
            columns={
                "sum": (
                    "FE_BUREAU_OVERDUE_ACCOUNT_COUNT"
                ),
                "mean": (
                    "FE_BUREAU_OVERDUE_ACCOUNT_RATIO"
                ),
            },
            inplace=True,
        )

        return result.merge(
            overdue_features,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    def _create_credit_history_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit-history duration features."""

        if "DAYS_CREDIT" not in dataframe.columns:
            return result

        history = dataframe.assign(
            _credit_age_years=(
                dataframe["DAYS_CREDIT"].abs()
                / 365.25
            ),
        )

        history_features = (
            history.groupby(
                APPLICATION_KEY,
                dropna=False,
            )["_credit_age_years"]
            .agg(
                [
                    "mean",
                    "min",
                    "max",
                ],
            )
            .reset_index()
        )

        history_features.rename(
            columns={
                "mean": (
                    "FE_BUREAU_CREDIT_AGE_MEAN_YEARS"
                ),
                "min": (
                    "FE_BUREAU_CREDIT_AGE_MIN_YEARS"
                ),
                "max": (
                    "FE_BUREAU_CREDIT_AGE_MAX_YEARS"
                ),
            },
            inplace=True,
        )

        return result.merge(
            history_features,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        """Safely divide two pandas Series."""

        denominator = denominator.replace(
            0,
            np.nan,
        )

        return numerator / denominator

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the bureau dataset."""

        if APPLICATION_KEY not in dataframe.columns:
            raise ValueError(
                f"Bureau dataset is missing "
                f"'{APPLICATION_KEY}'."
            )


bureau_feature_engineer = BureauFeatureEngineer()