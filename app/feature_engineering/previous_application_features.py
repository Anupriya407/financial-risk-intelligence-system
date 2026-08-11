"""
Previous application feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import APPLICATION_KEY


class PreviousApplicationFeatureEngineer(BaseFeatureEngineer):
    """Generate customer-level features from previous applications."""

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "previous_application"

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate customer-level previous-application features.

        Parameters
        ----------
        dataframe:
            Previous application dataset.

        Returns
        -------
        pandas.DataFrame
            One row per SK_ID_CURR.
        """

        self._validate_required_columns(dataframe)

        result = self._create_application_count_features(
            dataframe,
        )

        result = self._create_status_features(
            dataframe,
            result,
        )

        result = self._create_amount_features(
            dataframe,
            result,
        )

        result = self._create_timing_features(
            dataframe,
            result,
        )

        result.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return result

    def _create_application_count_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create previous application count features."""

        grouped = dataframe.groupby(
            APPLICATION_KEY,
            dropna=False,
        )

        result = (
            grouped.size()
            .reset_index(
                name="FE_PREV_APP_COUNT",
            )
        )

        if "NAME_CONTRACT_TYPE" in dataframe.columns:
            contract_type_count = (
                dataframe["NAME_CONTRACT_TYPE"]
                .notna()
                .groupby(
                    dataframe[APPLICATION_KEY],
                    dropna=False,
                )
                .sum()
                .reset_index(
                    name="FE_PREV_APP_CONTRACT_COUNT",
                )
            )

            result = result.merge(
                contract_type_count,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _create_status_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create approval and rejection features."""

        if "NAME_CONTRACT_STATUS" not in dataframe.columns:
            return result

        status = (
            dataframe["NAME_CONTRACT_STATUS"]
            .astype("string")
            .str.upper()
        )

        working = dataframe.assign(
            _approved=(
                status == "APPROVED"
            ).astype(int),
            _refused=(
                status == "REFUSED"
            ).astype(int),
            _cancelled=(
                status == "CANCELED"
            ).astype(int),
            _unused=(
                status == "UNUSED OFFER"
            ).astype(int),
        )

        status_features = (
            working.groupby(
                APPLICATION_KEY,
                dropna=False,
            )[
                [
                    "_approved",
                    "_refused",
                    "_cancelled",
                    "_unused",
                ]
            ]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "_approved": (
                        "FE_PREV_APP_APPROVED_COUNT"
                    ),
                    "_refused": (
                        "FE_PREV_APP_REFUSED_COUNT"
                    ),
                    "_cancelled": (
                        "FE_PREV_APP_CANCELLED_COUNT"
                    ),
                    "_unused": (
                        "FE_PREV_APP_UNUSED_COUNT"
                    ),
                }
            )
        )

        result = result.merge(
            status_features,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

        result["FE_PREV_APP_APPROVAL_RATIO"] = (
            result["FE_PREV_APP_APPROVED_COUNT"]
            / result["FE_PREV_APP_COUNT"]
        )

        result["FE_PREV_APP_REFUSAL_RATIO"] = (
            result["FE_PREV_APP_REFUSED_COUNT"]
            / result["FE_PREV_APP_COUNT"]
        )

        result["FE_PREV_APP_CANCELLATION_RATIO"] = (
            result["FE_PREV_APP_CANCELLED_COUNT"]
            / result["FE_PREV_APP_COUNT"]
        )

        return result

    def _create_amount_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create previous application amount features."""

        amount_columns = {
            "AMT_APPLICATION": [
                "sum",
                "mean",
                "max",
            ],
            "AMT_CREDIT": [
                "sum",
                "mean",
                "max",
            ],
            "AMT_ANNUITY": [
                "mean",
                "max",
            ],
            "AMT_DOWN_PAYMENT": [
                "mean",
                "max",
            ],
            "AMT_GOODS_PRICE": [
                "mean",
                "max",
            ],
        }

        available = {
            column: functions
            for column, functions in amount_columns.items()
            if column in dataframe.columns
        }

        if not available:
            return result

        grouped = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )
            .agg(available)
        )

        grouped.columns = [
            (
                f"FE_PREV_APP_"
                f"{column}_"
                f"{function}".upper()
            )
            for column, function in grouped.columns
        ]

        grouped = grouped.reset_index()

        result = result.merge(
            grouped,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

        if {
            "AMT_APPLICATION",
            "AMT_CREDIT",
        }.issubset(dataframe.columns):
            ratio_data = dataframe.assign(
                _application_credit_ratio=(
                    self._safe_divide(
                        dataframe["AMT_CREDIT"],
                        dataframe["AMT_APPLICATION"],
                    )
                )
            )

            ratio_features = (
                ratio_data.groupby(
                    APPLICATION_KEY,
                    dropna=False,
                )["_application_credit_ratio"]
                .agg(
                    [
                        "mean",
                        "max",
                    ]
                )
                .reset_index()
                .rename(
                    columns={
                        "mean": (
                            "FE_PREV_APP_CREDIT_APPLICATION_RATIO_MEAN"
                        ),
                        "max": (
                            "FE_PREV_APP_CREDIT_APPLICATION_RATIO_MAX"
                        ),
                    }
                )
            )

            result = result.merge(
                ratio_features,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _create_timing_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create previous application timing features."""

        timing_columns = {
            "DAYS_DECISION": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_FIRST_DRAWING": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_FIRST_DUE": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_LAST_DUE_1ST_VERSION": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_LAST_DUE": [
                "mean",
                "min",
                "max",
            ],
            "DAYS_TERMINATION": [
                "mean",
                "min",
                "max",
            ],
        }

        available = {
            column: functions
            for column, functions in timing_columns.items()
            if column in dataframe.columns
        }

        if not available:
            return result

        grouped = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )
            .agg(available)
        )

        grouped.columns = [
            (
                f"FE_PREV_APP_"
                f"{column}_"
                f"{function}".upper()
            )
            for column, function in grouped.columns
        ]

        grouped = grouped.reset_index()

        return result.merge(
            grouped,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the previous application dataset."""

        if APPLICATION_KEY not in dataframe.columns:
            raise ValueError(
                "Previous application dataset is "
                f"missing '{APPLICATION_KEY}'."
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


previous_application_feature_engineer = (
    PreviousApplicationFeatureEngineer()
)