"""
Installments payment feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import APPLICATION_KEY


class InstallmentsFeatureEngineer(BaseFeatureEngineer):
    """Generate customer-level features from installment payments."""

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "installments"

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate customer-level installment features.

        Parameters
        ----------
        dataframe:
            Installment payment dataset.

        Returns
        -------
        pandas.DataFrame
            One row per SK_ID_CURR.
        """

        self._validate_required_columns(
            dataframe,
        )

        working = self._create_row_level_features(
            dataframe,
        )

        result = self._create_payment_count_features(
            working,
        )

        result = self._create_amount_features(
            working,
            result,
        )

        result = self._create_payment_ratio_features(
            working,
            result,
        )

        result = self._create_delay_features(
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
        """Create features at individual installment level."""

        result = dataframe.copy(
            deep=True,
        )

        if {
            "AMT_PAYMENT",
            "AMT_INSTALMENT",
        }.issubset(result.columns):
            result["FE_INST_PAYMENT_RATIO"] = (
                self._safe_divide(
                    result["AMT_PAYMENT"],
                    result["AMT_INSTALMENT"],
                )
            )

            result["FE_INST_PAYMENT_GAP"] = (
                result["AMT_PAYMENT"]
                - result["AMT_INSTALMENT"]
            )

        if {
            "DAYS_ENTRY_PAYMENT",
            "DAYS_INSTALMENT",
        }.issubset(result.columns):
            result["FE_INST_PAYMENT_DELAY"] = (
                result["DAYS_ENTRY_PAYMENT"]
                - result["DAYS_INSTALMENT"]
            )

            result["FE_INST_LATE_PAYMENT"] = (
                result["FE_INST_PAYMENT_DELAY"]
                > 0
            ).astype(int)

            result["FE_INST_EARLY_PAYMENT"] = (
                result["FE_INST_PAYMENT_DELAY"]
                < 0
            ).astype(int)

            result["FE_INST_ON_TIME_PAYMENT"] = (
                result["FE_INST_PAYMENT_DELAY"]
                == 0
            ).astype(int)

        return result

    def _create_payment_count_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create installment count features."""

        grouped = dataframe.groupby(
            APPLICATION_KEY,
            dropna=False,
        )

        result = (
            grouped.size()
            .reset_index(
                name="FE_INST_PAYMENT_COUNT",
            )
        )

        if "SK_ID_PREV" in dataframe.columns:
            previous_count = (
                grouped["SK_ID_PREV"]
                .nunique()
                .reset_index(
                    name="FE_INST_PREVIOUS_LOAN_COUNT",
                )
            )

            result = result.merge(
                previous_count,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _create_amount_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create payment amount features."""

        amount_columns = {
            "AMT_PAYMENT": [
                "sum",
                "mean",
                "min",
                "max",
            ],
            "AMT_INSTALMENT": [
                "sum",
                "mean",
                "min",
                "max",
            ],
        }

        available = {
            column: functions
            for column, functions in amount_columns.items()
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
                    f"FE_INST_"
                    f"{column}_"
                    f"{function}".upper()
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

        return result

    def _create_payment_ratio_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create payment compliance ratio features."""

        if "FE_INST_PAYMENT_RATIO" not in dataframe.columns:
            return result

        ratio_features = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )["FE_INST_PAYMENT_RATIO"]
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
                        "FE_INST_PAYMENT_RATIO_MEAN"
                    ),
                    "min": (
                        "FE_INST_PAYMENT_RATIO_MIN"
                    ),
                    "max": (
                        "FE_INST_PAYMENT_RATIO_MAX"
                    ),
                }
            )
        )

        return result.merge(
            ratio_features,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    def _create_delay_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create payment delay features."""

        if "FE_INST_PAYMENT_DELAY" not in dataframe.columns:
            return result

        delay_features = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )["FE_INST_PAYMENT_DELAY"]
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
                        "FE_INST_PAYMENT_DELAY_MEAN"
                    ),
                    "min": (
                        "FE_INST_PAYMENT_DELAY_MIN"
                    ),
                    "max": (
                        "FE_INST_PAYMENT_DELAY_MAX"
                    ),
                }
            )
        )

        result = result.merge(
            delay_features,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

        if "FE_INST_LATE_PAYMENT" in dataframe.columns:
            late_features = (
                dataframe.groupby(
                    APPLICATION_KEY,
                    dropna=False,
                )["FE_INST_LATE_PAYMENT"]
                .agg(
                    [
                        "sum",
                        "mean",
                    ]
                )
                .reset_index()
                .rename(
                    columns={
                        "sum": (
                            "FE_INST_LATE_PAYMENT_COUNT"
                        ),
                        "mean": (
                            "FE_INST_LATE_PAYMENT_RATIO"
                        ),
                    }
                )
            )

            result = result.merge(
                late_features,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        if "FE_INST_EARLY_PAYMENT" in dataframe.columns:
            early_features = (
                dataframe.groupby(
                    APPLICATION_KEY,
                    dropna=False,
                )["FE_INST_EARLY_PAYMENT"]
                .agg(
                    [
                        "sum",
                        "mean",
                    ]
                )
                .reset_index()
                .rename(
                    columns={
                        "sum": (
                            "FE_INST_EARLY_PAYMENT_COUNT"
                        ),
                        "mean": (
                            "FE_INST_EARLY_PAYMENT_RATIO"
                        ),
                    }
                )
            )

            result = result.merge(
                early_features,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        if "FE_INST_ON_TIME_PAYMENT" in dataframe.columns:
            on_time_features = (
                dataframe.groupby(
                    APPLICATION_KEY,
                    dropna=False,
                )["FE_INST_ON_TIME_PAYMENT"]
                .agg(
                    [
                        "sum",
                        "mean",
                    ]
                )
                .reset_index()
                .rename(
                    columns={
                        "sum": (
                            "FE_INST_ON_TIME_PAYMENT_COUNT"
                        ),
                        "mean": (
                            "FE_INST_ON_TIME_PAYMENT_RATIO"
                        ),
                    }
                )
            )

            result = result.merge(
                on_time_features,
                on=APPLICATION_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the installment dataset."""

        if APPLICATION_KEY not in dataframe.columns:
            raise ValueError(
                "Installment dataset is missing "
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


installments_feature_engineer = (
    InstallmentsFeatureEngineer()
)