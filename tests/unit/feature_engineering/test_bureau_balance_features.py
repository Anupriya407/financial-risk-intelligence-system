"""
Tests for bureau balance feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.bureau_balance_features import (
    bureau_balance_feature_engineer,
)


def test_bureau_balance_features_create_customer_level_data() -> None:
    """Test bureau balance customer aggregation."""

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                1,
                1,
                2,
                2,
                3,
            ],
            "MONTHS_BALANCE": [
                -1,
                -2,
                -1,
                -2,
                -1,
            ],
            "STATUS": [
                "0",
                "1",
                "C",
                "0",
                "X",
            ],
        }
    )

    bureau = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                1,
                2,
                3,
            ],
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
        }
    )

    result = bureau_balance_feature_engineer.transform(
        bureau_balance,
        bureau,
    )

    assert len(result) == 2

    assert (
        "SK_ID_CURR"
        in result.columns
    )

    assert any(
        column.startswith(
            "FE_BUREAU_BALANCE_",
        )
        for column in result.columns
    )


def test_bureau_balance_features_have_unique_customers() -> None:
    """Test that output has one row per customer."""

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                1,
                1,
            ],
            "MONTHS_BALANCE": [
                -1,
                -2,
            ],
            "STATUS": [
                "0",
                "1",
            ],
        }
    )

    bureau = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                1,
            ],
            "SK_ID_CURR": [
                100001,
            ],
        }
    )

    result = bureau_balance_feature_engineer.transform(
        bureau_balance,
        bureau,
    )

    assert not result[
        "SK_ID_CURR"
    ].duplicated().any()


def test_bureau_balance_does_not_modify_input() -> None:
    """Test that the input DataFrame is unchanged."""

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                1,
            ],
            "MONTHS_BALANCE": [
                -1,
            ],
            "STATUS": [
                "0",
            ],
        }
    )

    bureau = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                1,
            ],
            "SK_ID_CURR": [
                100001,
            ],
        }
    )

    original_columns = (
        bureau_balance.columns.tolist()
    )

    bureau_balance_feature_engineer.transform(
        bureau_balance,
        bureau,
    )

    assert (
        bureau_balance.columns.tolist()
        == original_columns
    )