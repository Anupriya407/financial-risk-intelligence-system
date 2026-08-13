"""
Tests for credit card feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.credit_card_features import (
    credit_card_feature_engineer,
)


def test_credit_card_features_create_customer_level_data() -> None:
    """Test credit-card aggregation."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
            "SK_ID_PREV": [
                1,
                1,
                2,
            ],
            "MONTHS_BALANCE": [
                -1,
                -2,
                -1,
            ],
            "AMT_BALANCE": [
                50000.0,
                60000.0,
                30000.0,
            ],
            "AMT_CREDIT_LIMIT_ACTUAL": [
                100000.0,
                100000.0,
                100000.0,
            ],
            "AMT_PAYMENT_CURRENT": [
                5000.0,
                6000.0,
                3000.0,
            ],
            "AMT_PAYMENT_TOTAL_CURRENT": [
                5000.0,
                6000.0,
                3000.0,
            ],
            "AMT_INST_MIN_REGULARITY": [
                4000.0,
                4000.0,
                2500.0,
            ],
            "AMT_DRAWINGS_CURRENT": [
                10000.0,
                15000.0,
                5000.0,
            ],
            "CNT_DRAWINGS_CURRENT": [
                2,
                3,
                1,
            ],
        }
    )

    result = credit_card_feature_engineer.transform(
        dataframe,
    )

    assert len(result) == 2

    assert (
        "FE_CC_ACCOUNT_COUNT"
        in result.columns
    )

    assert (
        "FE_CC_AMT_BALANCE_MEAN"
        in result.columns
    )

    assert (
        "FE_CC_UTILIZATION_MEAN"
        in result.columns
    )


def test_credit_card_features_have_unique_customers() -> None:
    """Test one row per customer."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
            ],
        }
    )

    result = credit_card_feature_engineer.transform(
        dataframe,
    )

    assert not result[
        "SK_ID_CURR"
    ].duplicated().any()


def test_credit_card_features_do_not_modify_input() -> None:
    """Test that the input remains unchanged."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
            ],
        }
    )

    original_columns = dataframe.columns.tolist()

    credit_card_feature_engineer.transform(
        dataframe,
    )

    assert (
        dataframe.columns.tolist()
        == original_columns
    )