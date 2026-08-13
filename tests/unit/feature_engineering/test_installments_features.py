"""
Tests for installment payment feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.installments_features import (
    installments_feature_engineer,
)


def test_installment_features_create_customer_level_data() -> None:
    """Test installment aggregation."""

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
            "AMT_PAYMENT": [
                10000.0,
                8000.0,
                15000.0,
            ],
            "AMT_INSTALMENT": [
                10000.0,
                10000.0,
                15000.0,
            ],
            "DAYS_ENTRY_PAYMENT": [
                -10,
                5,
                -20,
            ],
            "DAYS_INSTALMENT": [
                -10,
                0,
                -20,
            ],
        }
    )

    result = installments_feature_engineer.transform(
        dataframe,
    )

    assert len(result) == 2

    assert (
        "FE_INST_PAYMENT_COUNT"
        in result.columns
    )

    assert (
        "FE_INST_PAYMENT_RATIO_MEAN"
        in result.columns
    )

    assert (
        "FE_INST_LATE_PAYMENT_COUNT"
        in result.columns
    )

    assert (
        "FE_INST_LATE_PAYMENT_RATIO"
        in result.columns
    )


def test_installment_features_have_unique_customers() -> None:
    """Test one row per customer."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
            ],
        }
    )

    result = installments_feature_engineer.transform(
        dataframe,
    )

    assert not result[
        "SK_ID_CURR"
    ].duplicated().any()


def test_installment_features_do_not_modify_input() -> None:
    """Test that the input remains unchanged."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
            ],
        }
    )

    original_columns = dataframe.columns.tolist()

    installments_feature_engineer.transform(
        dataframe,
    )

    assert (
        dataframe.columns.tolist()
        == original_columns
    )