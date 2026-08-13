"""
Tests for previous application feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.previous_application_features import (
    previous_application_feature_engineer,
)


def test_previous_application_features_create_customer_level_data() -> None:
    """Test previous application aggregation."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
            "NAME_CONTRACT_STATUS": [
                "Approved",
                "Refused",
                "Approved",
            ],
            "AMT_APPLICATION": [
                100000.0,
                200000.0,
                300000.0,
            ],
            "AMT_CREDIT": [
                90000.0,
                0.0,
                280000.0,
            ],
            "AMT_ANNUITY": [
                5000.0,
                0.0,
                12000.0,
            ],
            "AMT_DOWN_PAYMENT": [
                10000.0,
                0.0,
                20000.0,
            ],
            "AMT_GOODS_PRICE": [
                100000.0,
                200000.0,
                300000.0,
            ],
            "DAYS_DECISION": [
                -100,
                -200,
                -50,
            ],
        }
    )

    result = previous_application_feature_engineer.transform(
        dataframe,
    )

    assert len(result) == 2

    assert (
        "FE_PREV_APP_COUNT"
        in result.columns
    )

    assert (
        "FE_PREV_APP_APPROVED_COUNT"
        in result.columns
    )

    assert (
        "FE_PREV_APP_REFUSED_COUNT"
        in result.columns
    )

    assert (
        "FE_PREV_APP_APPROVAL_RATIO"
        in result.columns
    )


def test_previous_application_features_have_unique_customers() -> None:
    """Test one row per customer."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
            ],
        }
    )

    result = previous_application_feature_engineer.transform(
        dataframe,
    )

    assert not result[
        "SK_ID_CURR"
    ].duplicated().any()


def test_previous_application_does_not_modify_input() -> None:
    """Test that the input remains unchanged."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
            ],
        }
    )

    original_columns = dataframe.columns.tolist()

    previous_application_feature_engineer.transform(
        dataframe,
    )

    assert (
        dataframe.columns.tolist()
        == original_columns
    )