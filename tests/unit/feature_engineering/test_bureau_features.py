"""
Tests for bureau feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.bureau_features import (
    bureau_feature_engineer,
)


def test_bureau_features_create_customer_level_data() -> None:
    """Test bureau aggregation."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
            "AMT_CREDIT_SUM": [
                100000.0,
                200000.0,
                300000.0,
            ],
            "AMT_CREDIT_SUM_DEBT": [
                50000.0,
                100000.0,
                150000.0,
            ],
            "AMT_CREDIT_SUM_OVERDUE": [
                0.0,
                5000.0,
                0.0,
            ],
            "AMT_CREDIT_MAX_OVERDUE": [
                0.0,
                5000.0,
                0.0,
            ],
            "DAYS_CREDIT": [
                -1000,
                -2000,
                -500,
            ],
            "DAYS_CREDIT_ENDDATE": [
                100,
                200,
                300,
            ],
            "DAYS_ENDDATE_FACT": [
                -50,
                -100,
                -200,
            ],
            "CNT_CREDIT_PROLONG": [
                0,
                1,
                0,
            ],
            "CREDIT_ACTIVE": [
                "Active",
                "Closed",
                "Active",
            ],
        }
    )

    result = bureau_feature_engineer.transform(
        dataframe,
    )

    assert len(result) == 2

    assert (
        "FE_BUREAU_ACCOUNT_COUNT"
        in result.columns
    )

    assert (
        "FE_BUREAU_ACTIVE_COUNT"
        in result.columns
    )

    assert (
        "FE_BUREAU_OVERDUE_ACCOUNT_COUNT"
        in result.columns
    )

    assert (
        "FE_BUREAU_CREDIT_AGE_MEAN_YEARS"
        in result.columns
    )


def test_bureau_features_have_unique_customer_keys() -> None:
    """Test one row per customer."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
        }
    )

    result = bureau_feature_engineer.transform(
        dataframe,
    )

    assert not result[
        "SK_ID_CURR"
    ].duplicated().any()


def test_bureau_features_do_not_modify_input() -> None:
    """Test input DataFrame remains unchanged."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
        }
    )

    original_columns = dataframe.columns.tolist()

    bureau_feature_engineer.transform(
        dataframe,
    )

    assert (
        dataframe.columns.tolist()
        == original_columns
    )