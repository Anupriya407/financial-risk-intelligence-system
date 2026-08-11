"""
Tests for POS Cash feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.pos_cash_features import (
    pos_cash_feature_engineer,
)


def test_pos_cash_features_create_customer_level_data() -> None:
    """Test POS Cash aggregation."""

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
            "NAME_CONTRACT_STATUS": [
                "Active",
                "Completed",
                "Active",
            ],
            "CNT_INSTALMENT": [
                12,
                12,
                24,
            ],
            "CNT_INSTALMENT_FUTURE": [
                6,
                5,
                20,
            ],
            "SK_DPD": [
                0,
                5,
                2,
            ],
            "SK_DPD_DEF": [
                0,
                1,
                0,
            ],
        }
    )

    result = pos_cash_feature_engineer.transform(
        dataframe,
    )

    assert len(result) == 2

    assert (
        "FE_POS_RECORD_COUNT"
        in result.columns
    )

    assert (
        "FE_POS_PREVIOUS_LOAN_COUNT"
        in result.columns
    )

    assert (
        "FE_POS_COMPLETED_COUNT"
        in result.columns
    )

    assert (
        "FE_POS_COMPLETED_RATIO"
        in result.columns
    )

    assert (
        "FE_POS_SK_DPD_MEAN"
        in result.columns
    )


def test_pos_cash_features_have_unique_customers() -> None:
    """Test one row per customer."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
            ],
        }
    )

    result = pos_cash_feature_engineer.transform(
        dataframe,
    )

    assert not result[
        "SK_ID_CURR"
    ].duplicated().any()


def test_pos_cash_features_do_not_modify_input() -> None:
    """Test that the input remains unchanged."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
            ],
        }
    )

    original_columns = dataframe.columns.tolist()

    pos_cash_feature_engineer.transform(
        dataframe,
    )

    assert (
        dataframe.columns.tolist()
        == original_columns
    )