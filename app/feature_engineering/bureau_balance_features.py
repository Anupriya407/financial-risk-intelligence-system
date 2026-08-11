"""
Bureau balance feature engineering for the
Financial Risk Intelligence System.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_engineering.base import BaseFeatureEngineer
from app.feature_engineering.config import (
    APPLICATION_KEY,
    BUREAU_KEY,
)


class BureauBalanceFeatureEngineer(BaseFeatureEngineer):
    """
    Generate customer-level features from monthly
    bureau balance history.
    """

    @property
    def name(self) -> str:
        """Return the feature engineer name."""

        return "bureau_balance"

    def transform(
        self,
        dataframe: pd.DataFrame,
        bureau_dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate customer-level bureau balance features.

        Parameters
        ----------
        dataframe:
            bureau_balance dataset.

        bureau_dataframe:
            Bureau dataset used to map SK_ID_BUREAU
            to SK_ID_CURR.

        Returns
        -------
        pandas.DataFrame
            Customer-level bureau balance features.
        """

        self._validate_required_columns(
            dataframe,
        )

        if bureau_dataframe is None:
            raise ValueError(
                "bureau_dataframe is required to map "
                "SK_ID_BUREAU to SK_ID_CURR."
            )

        self._validate_bureau_mapping(
            bureau_dataframe,
        )

        bureau_features = self._create_bureau_level_features(
            dataframe,
        )

        customer_features = self._merge_with_bureau(
            bureau_features,
            bureau_dataframe,
        )

        customer_features = (
            self._create_customer_level_features(
                customer_features,
            )
        )

        customer_features.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return customer_features

    def _create_bureau_level_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate monthly records to SK_ID_BUREAU level.
        """

        grouped = dataframe.groupby(
            BUREAU_KEY,
            dropna=False,
        )

        result = grouped.size().reset_index(
            name="FE_BUREAU_BALANCE_MONTH_COUNT",
        )

        if "MONTHS_BALANCE" in dataframe.columns:
            months_features = (
                grouped["MONTHS_BALANCE"]
                .agg(
                    [
                        "min",
                        "max",
                    ],
                )
                .reset_index()
            )

            months_features.rename(
                columns={
                    "min": (
                        "FE_BUREAU_BALANCE_MONTHS_MIN"
                    ),
                    "max": (
                        "FE_BUREAU_BALANCE_MONTHS_MAX"
                    ),
                },
                inplace=True,
            )

            result = result.merge(
                months_features,
                on=BUREAU_KEY,
                how="left",
                validate="one_to_one",
            )

        if "STATUS" in dataframe.columns:
            status_features = self._create_status_features(
                dataframe,
            )

            result = result.merge(
                status_features,
                on=BUREAU_KEY,
                how="left",
                validate="one_to_one",
            )

        return result

    def _create_status_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create monthly credit-status features."""

        working = dataframe.copy(
            deep=True,
        )

        status = (
            working["STATUS"]
            .astype("string")
            .str.upper()
        )

        working["_status_c"] = (
            status == "C"
        ).astype(int)

        working["_status_0"] = (
            status == "0"
        ).astype(int)

        working["_status_1"] = (
            status == "1"
        ).astype(int)

        working["_status_2"] = (
            status == "2"
        ).astype(int)

        working["_status_3"] = (
            status == "3"
        ).astype(int)

        working["_status_4"] = (
            status == "4"
        ).astype(int)

        working["_status_5"] = (
            status == "5"
        ).astype(int)

        working["_status_x"] = (
            status == "X"
        ).astype(int)

        result = (
            working.groupby(
                BUREAU_KEY,
                dropna=False,
            )[
                [
                    "_status_c",
                    "_status_0",
                    "_status_1",
                    "_status_2",
                    "_status_3",
                    "_status_4",
                    "_status_5",
                    "_status_x",
                ]
            ]
            .agg(
                [
                    "sum",
                    "mean",
                ],
            )
        )

        result.columns = [
            "FE_BUREAU_BALANCE_STATUS_"
            f"{status_name.upper()}_"
            f"{aggregation.upper()}"
            for status_name, aggregation
            in result.columns
        ]

        return result.reset_index()

    def _merge_with_bureau(
        self,
        bureau_balance_features: pd.DataFrame,
        bureau_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach SK_ID_CURR to bureau-balance features.
        """

        mapping = bureau_dataframe[
            [
                BUREAU_KEY,
                APPLICATION_KEY,
            ]
        ].drop_duplicates()

        duplicate_bureau_ids = mapping.duplicated(
            subset=[BUREAU_KEY],
        )

        if duplicate_bureau_ids.any():
            raise ValueError(
                "Bureau dataset contains duplicate "
                "SK_ID_BUREAU values."
            )

        return bureau_balance_features.merge(
            mapping,
            on=BUREAU_KEY,
            how="inner",
            validate="one_to_one",
        )

    def _create_customer_level_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate bureau-balance features to customer level."""

        feature_columns = [
            column
            for column in dataframe.columns
            if column.startswith(
                "FE_BUREAU_BALANCE_",
            )
            and column != BUREAU_KEY
        ]

        if not feature_columns:
            return (
                dataframe[
                    [APPLICATION_KEY]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        aggregations = {
            column: [
                "sum",
                "mean",
                "max",
            ]
            for column in feature_columns
        }

        result = (
            dataframe.groupby(
                APPLICATION_KEY,
                dropna=False,
            )[feature_columns]
            .agg(aggregations)
        )

        result.columns = [
            (
                f"FE_BUREAU_BALANCE_CUSTOMER_"
                f"{column}_{aggregation}"
            )
            for column, aggregation
            in result.columns
        ]

        return result.reset_index()

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate required bureau balance columns."""

        required_columns = [
            BUREAU_KEY,
        ]

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Bureau balance dataset is missing "
                f"required columns: {missing}"
            )

    def _validate_bureau_mapping(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the bureau-to-customer mapping."""

        required_columns = [
            BUREAU_KEY,
            APPLICATION_KEY,
        ]

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Bureau dataset is missing mapping "
                f"columns: {missing}"
            )


bureau_balance_feature_engineer = (
    BureauBalanceFeatureEngineer()
)