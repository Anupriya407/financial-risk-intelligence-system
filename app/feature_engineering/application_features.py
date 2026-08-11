"""
Application-level feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import APPLICATION_KEY


class ApplicationFeatureEngineer(BaseFeatureEngineer):
    """Generate features from the application dataset."""

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "application"

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate application-level financial risk features.

        Parameters
        ----------
        dataframe:
            Application dataset.

        Returns
        -------
        pandas.DataFrame
            Application dataset with engineered features.
        """

        result = dataframe.copy(
            deep=True,
        )

        self._validate_required_columns(
            result,
        )

        # ---------------------------------------------------------------
        # Financial Ratio Features
        # ---------------------------------------------------------------

        result["FE_CREDIT_INCOME_RATIO"] = (
            self._safe_divide(
                result["AMT_CREDIT"],
                result["AMT_INCOME_TOTAL"],
            )
        )

        result["FE_ANNUITY_INCOME_RATIO"] = (
            self._safe_divide(
                result["AMT_ANNUITY"],
                result["AMT_INCOME_TOTAL"],
            )
        )

        result["FE_GOODS_INCOME_RATIO"] = (
            self._safe_divide(
                result["AMT_GOODS_PRICE"],
                result["AMT_INCOME_TOTAL"],
            )
        )

        result["FE_CREDIT_GOODS_RATIO"] = (
            self._safe_divide(
                result["AMT_CREDIT"],
                result["AMT_GOODS_PRICE"],
            )
        )

        result["FE_ANNUITY_CREDIT_RATIO"] = (
            self._safe_divide(
                result["AMT_ANNUITY"],
                result["AMT_CREDIT"],
            )
        )

        # ---------------------------------------------------------------
        # Income / Family Features
        # ---------------------------------------------------------------

        result["FE_INCOME_PER_PERSON"] = (
            self._safe_divide(
                result["AMT_INCOME_TOTAL"],
                result["CNT_FAM_MEMBERS"],
            )
        )

        result["FE_CHILDREN_RATIO"] = (
            self._safe_divide(
                result["CNT_CHILDREN"],
                result["CNT_FAM_MEMBERS"],
            )
        )

        result["FE_INCOME_PER_CHILD"] = (
            self._safe_divide(
                result["AMT_INCOME_TOTAL"],
                result["CNT_CHILDREN"],
            )
        )

        # ---------------------------------------------------------------
        # Age Features
        # ---------------------------------------------------------------

        result["FE_AGE_YEARS"] = (
            self._days_to_years(
                result["DAYS_BIRTH"],
            )
        )

        result["FE_EMPLOYMENT_YEARS"] = (
            self._days_to_years(
                result["DAYS_EMPLOYED"],
            )
        )

        result["FE_REGISTRATION_YEARS"] = (
            self._days_to_years(
                result["DAYS_REGISTRATION"],
            )
        )

        result["FE_ID_PUBLISH_YEARS"] = (
            self._days_to_years(
                result["DAYS_ID_PUBLISH"],
            )
        )

        # ---------------------------------------------------------------
        # Employment / Age Relationship
        # ---------------------------------------------------------------

        result["FE_EMPLOYMENT_AGE_RATIO"] = (
            self._safe_divide(
                result["FE_EMPLOYMENT_YEARS"],
                result["FE_AGE_YEARS"],
            )
        )

        result["FE_INCOME_AGE_RATIO"] = (
            self._safe_divide(
                result["AMT_INCOME_TOTAL"],
                result["FE_AGE_YEARS"],
            )
        )

        # ---------------------------------------------------------------
        # External Risk Source Features
        # ---------------------------------------------------------------

        external_sources = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ]

        available_external_sources = [
            column
            for column in external_sources
            if column in result.columns
        ]

        if available_external_sources:
            result["FE_EXT_SOURCE_MEAN"] = (
                result[
                    available_external_sources
                ].mean(axis=1)
            )

            result["FE_EXT_SOURCE_MIN"] = (
                result[
                    available_external_sources
                ].min(axis=1)
            )

            result["FE_EXT_SOURCE_MAX"] = (
                result[
                    available_external_sources
                ].max(axis=1)
            )

            result["FE_EXT_SOURCE_STD"] = (
                result[
                    available_external_sources
                ].std(axis=1)
            )

        # ---------------------------------------------------------------
        # Document / Contact Indicators
        # ---------------------------------------------------------------

        document_columns = [
            column
            for column in result.columns
            if column.startswith("FLAG_DOCUMENT_")
        ]

        if document_columns:
            result["FE_DOCUMENT_COUNT"] = (
                result[
                    document_columns
                ].sum(axis=1)
            )

        contact_columns = [
            column
            for column in (
                "FLAG_MOBIL",
                "FLAG_EMP_PHONE",
                "FLAG_WORK_PHONE",
                "FLAG_CONT_MOBILE",
                "FLAG_PHONE",
                "FLAG_EMAIL",
            )
            if column in result.columns
        ]

        if contact_columns:
            result["FE_CONTACT_FLAG_COUNT"] = (
                result[
                    contact_columns
                ].sum(axis=1)
            )

        # ---------------------------------------------------------------
        # Housing / Family Indicators
        # ---------------------------------------------------------------

        if {
            "OWN_CAR_AGE",
            "FLAG_OWN_CAR",
        }.issubset(result.columns):
            result["FE_CAR_OWNERSHIP_AGE"] = np.where(
                result["FLAG_OWN_CAR"] == "Y",
                result["OWN_CAR_AGE"],
                np.nan,
            )

        # ---------------------------------------------------------------
        # Credit / Income Difference Features
        # ---------------------------------------------------------------

        result["FE_INCOME_CREDIT_GAP"] = (
            result["AMT_INCOME_TOTAL"]
            - result["AMT_CREDIT"]
        )

        result["FE_INCOME_ANNUITY_GAP"] = (
            result["AMT_INCOME_TOTAL"]
            - result["AMT_ANNUITY"]
        )

        result["FE_CREDIT_GOODS_GAP"] = (
            result["AMT_CREDIT"]
            - result["AMT_GOODS_PRICE"]
        )

        # ---------------------------------------------------------------
        # Sanity Cleanup
        # ---------------------------------------------------------------

        result.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return result

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the minimum application columns."""

        required_columns = [
            APPLICATION_KEY,
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
        ]

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Application dataset is missing "
                f"required columns: {missing}"
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

    @staticmethod
    def _days_to_years(
        days: pd.Series,
    ) -> pd.Series:
        """
        Convert Home Credit negative day values
        into positive years.
        """

        return days.abs() / 365.25


application_feature_engineer = (
    ApplicationFeatureEngineer()
)