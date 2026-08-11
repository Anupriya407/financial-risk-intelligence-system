"""
Credit card balance feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import APPLICATION_KEY


class CreditCardFeatureEngineer(BaseFeatureEngineer):
    """Generate customer-level features from credit card history."""

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "credit_card"

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate customer-level credit card features.

        Parameters
        ----------
        dataframe:
            Credit card balance dataset.

        Returns
        -------
        pandas.DataFrame
            One row per SK_ID_CURR.
        """

        self._validate_required_columns(dataframe)

        working = self._create_row_level_features(
            dataframe,
        )

        result = self._create_account_features(
            working,
        )

        result = self._create_balance_features(
            working,
            result,
        )

        result = self._create_payment_features(
            working,
            result,
        )

        result = self._create_utilization_features(
            working,
            result,
        )

        result = self._create_activity_features(
            working,
            result,
        )

        result.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return result

    def _create_row_level_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit-card record-level features."""

        result = dataframe.copy(
            deep=True,
        )

        if {
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
        }.issubset(result.columns):
            result["FE_CC_UTILIZATION"] = (
                self._safe_divide(
                    result["AMT_BALANCE"],
                    result["AMT_CREDIT_LIMIT_ACTUAL"],
                )
            )

        if {
            "AMT_PAYMENT_TOTAL_CURRENT",
            "AMT_INST_MIN_REGULARITY",
        }.issubset(result.columns):
            result["FE_CC_PAYMENT_MIN_RATIO"] = (
                self._safe_divide(
                    result["AMT_PAYMENT_TOTAL_CURRENT"],
                    result["AMT_INST_MIN_REGULARITY"],
                )
            )

        if {
            "AMT_PAYMENT_CURRENT",
            "AMT_INST_MIN_REGULARITY",
        }.issubset(result.columns):
            result["FE_CC_CURRENT_PAYMENT_MIN_RATIO"] = (
                self._safe_divide(
                    result["AMT_PAYMENT_CURRENT"],
                    result["AMT_INST_MIN_REGULARITY"],
                )
            )

        if {
            "AMT_DRAWINGS_CURRENT",
            "AMT_CREDIT_LIMIT_ACTUAL",
        }.issubset(result.columns):
            result["FE_CC_DRAWING_LIMIT_RATIO"] = (
                self._safe_divide(
                    result["AMT_DRAWINGS_CURRENT"],
                    result["AMT_CREDIT_LIMIT_ACTUAL"],
                )
            )

        return result

    def _create_account_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit-card account count features."""

        grouped = dataframe.groupby(
            APPLICATION_KEY,
            dropna=False,
        )

        result = (
            grouped.size()
            .reset_index(
                name="FE_CC_RECORD_COUNT",
            )
        )

        if "SK_ID_PREV" in dataframe.columns:
            account_count = (
                grouped["SK_ID_PREV"]
                .nunique()
                .reset_index(
                    name="FE_CC_ACCOUNT_COUNT",
                )
            )

            result = result.merge(
                account_count,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        if "MONTHS_BALANCE" in dataframe.columns:
            month_features = (
                grouped["MONTHS_BALANCE"]
                .agg(
                    [
                        "min",
                        "max",
                    ]
                )
                .reset_index()
                .rename(
                    columns={
                        "min": (
                            "FE_CC_MONTHS_BALANCE_MIN"
                        ),
                        "max": (
                            "FE_CC_MONTHS_BALANCE_MAX"
                        ),
                    }
                )
            )

            result = result.merge(
                month_features,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _create_balance_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit-card balance features."""

        balance_columns = {
            "AMT_BALANCE": [
                "mean",
                "min",
                "max",
                "sum",
            ],
            "AMT_CREDIT_LIMIT_ACTUAL": [
                "mean",
                "max",
            ],
            "AMT_RECEIVABLE_PRINCIPAL": [
                "mean",
                "max",
            ],
            "AMT_TOTAL_RECEIVABLE": [
                "mean",
                "max",
            ],
        }

        available = {
            column: functions
            for column, functions
            in balance_columns.items()
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
                f"FE_CC_{column}_{function}".upper()
            )
            for column, function
            in grouped.columns
        ]

        grouped = grouped.reset_index()

        return result.merge(
            grouped,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    def _create_payment_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit-card payment features."""

        payment_columns = {
            "AMT_PAYMENT_CURRENT": [
                "mean",
                "max",
                "sum",
            ],
            "AMT_PAYMENT_TOTAL_CURRENT": [
                "mean",
                "max",
                "sum",
            ],
            "AMT_INST_MIN_REGULARITY": [
                "mean",
                "max",
            ],
        }

        available = {
            column: functions
            for column, functions
            in payment_columns.items()
            if column in dataframe.columns
        }

        if available:
            grouped = (
                dataframe.groupby(
                    APPLICATION_KEY,
                    dropna=False,
                )
                .agg(available)
            )

            grouped.columns = [
                (
                    f"FE_CC_{column}_{function}".upper()
                )
                for column, function
                in grouped.columns
            ]

            grouped = grouped.reset_index()

            result = result.merge(
                grouped,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        ratio_columns = [
            column
            for column in dataframe.columns
            if column.startswith(
                "FE_CC_"
            )
            and (
                "RATIO" in column
            )
        ]

        if ratio_columns:
            ratio_features = (
                dataframe.groupby(
                    APPLICATION_KEY,
                    dropna=False,
                )[ratio_columns]
                .agg(
                    [
                        "mean",
                        "max",
                    ]
                )
            )

            ratio_features.columns = [
                (
                    f"{column}_{function}".upper()
                )
                for column, function
                in ratio_features.columns
            ]

            ratio_features = (
                ratio_features.reset_index()
            )

            result = result.merge(
                ratio_features,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _create_utilization_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit utilization features."""

        if "FE_CC_UTILIZATION" not in dataframe.columns:
            return result

        utilization = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )["FE_CC_UTILIZATION"]
            .agg(
                [
                    "mean",
                    "min",
                    "max",
                ]
            )
            .reset_index()
            .rename(
                columns={
                    "mean": (
                        "FE_CC_UTILIZATION_MEAN"
                    ),
                    "min": (
                        "FE_CC_UTILIZATION_MIN"
                    ),
                    "max": (
                        "FE_CC_UTILIZATION_MAX"
                    ),
                }
            )
        )

        return result.merge(
            utilization,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    def _create_activity_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create credit-card activity features."""

        activity_columns = {
            "AMT_DRAWINGS_CURRENT": [
                "mean",
                "max",
                "sum",
            ],
            "CNT_DRAWINGS_CURRENT": [
                "mean",
                "max",
                "sum",
            ],
            "CNT_DRAWINGS_ATM_CURRENT": [
                "mean",
                "max",
                "sum",
            ],
            "CNT_DRAWINGS_POS_CURRENT": [
                "mean",
                "max",
                "sum",
            ],
            "CNT_INSTALMENT_MATURE_CUM": [
                "mean",
                "max",
            ],
        }

        available = {
            column: functions
            for column, functions
            in activity_columns.items()
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
                f"FE_CC_{column}_{function}".upper()
            )
            for column, function
            in grouped.columns
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
        """Validate the credit-card dataset."""

        if APPLICATION_KEY not in dataframe.columns:
            raise ValueError(
                "Credit card dataset is missing "
                f"'{APPLICATION_KEY}'."
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


credit_card_feature_engineer = (
    CreditCardFeatureEngineer()
)