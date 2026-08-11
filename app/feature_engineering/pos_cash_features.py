"""
POS Cash balance feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import APPLICATION_KEY


class PosCashFeatureEngineer(BaseFeatureEngineer):
    """Generate customer-level features from POS Cash history."""

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "pos_cash"

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate customer-level POS Cash features.

        Parameters
        ----------
        dataframe:
            POS Cash balance dataset.

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

        result = self._create_count_features(
            working,
        )

        result = self._create_status_features(
            working,
            result,
        )

        result = self._create_installment_features(
            working,
            result,
        )

        result = self._create_timing_features(
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
        """Create POS Cash record-level features."""

        result = dataframe.copy(
            deep=True,
        )

        if {
            "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
        }.issubset(result.columns):
            result["FE_POS_REMAINING_INSTALLMENT_RATIO"] = (
                self._safe_divide(
                    result["CNT_INSTALMENT_FUTURE"],
                    result["CNT_INSTALMENT"],
                )
            )

            result["FE_POS_COMPLETED_INSTALLMENT_RATIO"] = (
                1
                - result[
                    "FE_POS_REMAINING_INSTALLMENT_RATIO"
                ]
            )

        if {
            "SK_DPD",
            "SK_DPD_DEF",
        }.issubset(result.columns):
            result["FE_POS_DPD_GAP"] = (
                result["SK_DPD"]
                - result["SK_DPD_DEF"]
            )

        if "SK_DPD" in result.columns:
            result["FE_POS_LATE_PAYMENT"] = (
                result["SK_DPD"] > 0
            ).astype(int)

        if "SK_DPD_DEF" in result.columns:
            result["FE_POS_DEFAULT_DPD"] = (
                result["SK_DPD_DEF"] > 0
            ).astype(int)

        return result

    def _create_count_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create POS Cash count features."""

        grouped = dataframe.groupby(
            APPLICATION_KEY,
            dropna=False,
        )

        result = (
            grouped.size()
            .reset_index(
                name="FE_POS_RECORD_COUNT",
            )
        )

        if "SK_ID_PREV" in dataframe.columns:
            previous_count = (
                grouped["SK_ID_PREV"]
                .nunique()
                .reset_index(
                    name="FE_POS_PREVIOUS_LOAN_COUNT",
                )
            )

            result = result.merge(
                previous_count,
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
        """Create contract-status features."""

        if "NAME_CONTRACT_STATUS" not in dataframe.columns:
            return result

        status = (
            dataframe["NAME_CONTRACT_STATUS"]
            .astype("string")
            .str.upper()
        )

        working = dataframe.assign(
            _completed=(
                status == "COMPLETED"
            ).astype(int),
            _active=(
                status == "ACTIVE"
            ).astype(int),
            _signed=(
                status == "SIGNED"
            ).astype(int),
            _returned=(
                status == "RETURNED TO STORE"
            ).astype(int),
        )

        status_features = (
            working.groupby(
                APPLICATION_KEY,
                dropna=False,
            )[
                [
                    "_completed",
                    "_active",
                    "_signed",
                    "_returned",
                ]
            ]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "_completed": (
                        "FE_POS_COMPLETED_COUNT"
                    ),
                    "_active": (
                        "FE_POS_ACTIVE_COUNT"
                    ),
                    "_signed": (
                        "FE_POS_SIGNED_COUNT"
                    ),
                    "_returned": (
                        "FE_POS_RETURNED_COUNT"
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

        result["FE_POS_COMPLETED_RATIO"] = (
            self._safe_divide(
                result["FE_POS_COMPLETED_COUNT"],
                result["FE_POS_RECORD_COUNT"],
            )
        )

        result["FE_POS_ACTIVE_RATIO"] = (
            self._safe_divide(
                result["FE_POS_ACTIVE_COUNT"],
                result["FE_POS_RECORD_COUNT"],
            )
        )

        return result

    def _create_installment_features(
        self,
        dataframe: pd.DataFrame,
        result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create installment-related features."""

        installment_columns = {
            "CNT_INSTALMENT": [
                "mean",
                "min",
                "max",
            ],
            "CNT_INSTALMENT_FUTURE": [
                "mean",
                "min",
                "max",
            ],
            "SK_DPD": [
                "mean",
                "min",
                "max",
                "sum",
            ],
            "SK_DPD_DEF": [
                "mean",
                "min",
                "max",
                "sum",
            ],
        }

        available = {
            column: functions
            for column, functions
            in installment_columns.items()
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
                    f"FE_POS_{column}_{function}".upper()
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
                "FE_POS_"
            )
            and "RATIO" in column
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

            result = result.merge(
                ratio_features.reset_index(),
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
        """Create POS Cash temporal features."""

        if "MONTHS_BALANCE" not in dataframe.columns:
            return result

        timing_features = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )["MONTHS_BALANCE"]
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
                        "FE_POS_MONTHS_BALANCE_MIN"
                    ),
                    "max": (
                        "FE_POS_MONTHS_BALANCE_MAX"
                    ),
                }
            )
        )

        timing_features["FE_POS_HISTORY_LENGTH"] = (
            timing_features[
                "FE_POS_MONTHS_BALANCE_MIN"
            ]
            - timing_features[
                "FE_POS_MONTHS_BALANCE_MAX"
            ]
            .abs()
        )

        return result.merge(
            timing_features,
            on=APPLICATION_KEY,
            how="left",
            validate="one_to_one",
        )

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the POS Cash dataset."""

        if APPLICATION_KEY not in dataframe.columns:
            raise ValueError(
                "POS Cash dataset is missing "
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


pos_cash_feature_engineer = (
    PosCashFeatureEngineer()
)