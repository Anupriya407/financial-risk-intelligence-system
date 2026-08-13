"""
Tests for application-level feature engineering.
"""

from __future__ import annotations

import pandas as pd

from app.feature_engineering.application_features import (
    application_feature_engineer,
)


def test_application_feature_engineer_creates_features() -> None:
    """Test application feature generation."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0],
            "AMT_CREDIT": [200000.0, 300000.0],
            "AMT_ANNUITY": [10000.0, 15000.0],
            "AMT_GOODS_PRICE": [180000.0, 280000.0],
            "CNT_CHILDREN": [1, 2],
            "CNT_FAM_MEMBERS": [3, 4],
            "DAYS_BIRTH": [-10000, -15000],
            "DAYS_EMPLOYED": [-2000, -5000],
            "DAYS_REGISTRATION": [-3000, -4000],
            "DAYS_ID_PUBLISH": [-1000, -2000],
        }
    )

    result = application_feature_engineer.transform(
        dataframe,
    )

    expected_features = [
        "FE_CREDIT_INCOME_RATIO",
        "FE_ANNUITY_INCOME_RATIO",
        "FE_GOODS_INCOME_RATIO",
        "FE_CREDIT_GOODS_RATIO",
        "FE_ANNUITY_CREDIT_RATIO",
        "FE_INCOME_PER_PERSON",
        "FE_CHILDREN_RATIO",
        "FE_INCOME_PER_CHILD",
        "FE_AGE_YEARS",
        "FE_EMPLOYMENT_YEARS",
        "FE_REGISTRATION_YEARS",
        "FE_ID_PUBLISH_YEARS",
        "FE_EMPLOYMENT_AGE_RATIO",
        "FE_INCOME_AGE_RATIO",
        "FE_INCOME_CREDIT_GAP",
        "FE_INCOME_ANNUITY_GAP",
        "FE_CREDIT_GOODS_GAP",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_application_feature_engineer_preserves_rows() -> None:
    """Test that feature engineering preserves row count."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0],
            "AMT_CREDIT": [200000.0, 300000.0],
            "AMT_ANNUITY": [10000.0, 15000.0],
            "AMT_GOODS_PRICE": [180000.0, 280000.0],
            "CNT_CHILDREN": [1, 2],
            "CNT_FAM_MEMBERS": [3, 4],
            "DAYS_BIRTH": [-10000, -15000],
            "DAYS_EMPLOYED": [-2000, -5000],
            "DAYS_REGISTRATION": [-3000, -4000],
            "DAYS_ID_PUBLISH": [-1000, -2000],
        }
    )

    result = application_feature_engineer.transform(
        dataframe,
    )

    assert len(result) == len(dataframe)


def test_application_feature_engineer_does_not_modify_input() -> None:
    """Test that the original DataFrame remains unchanged."""

    dataframe = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_CREDIT": [200000.0],
            "AMT_ANNUITY": [10000.0],
            "AMT_GOODS_PRICE": [180000.0],
            "CNT_CHILDREN": [1],
            "CNT_FAM_MEMBERS": [3],
            "DAYS_BIRTH": [-10000],
            "DAYS_EMPLOYED": [-2000],
            "DAYS_REGISTRATION": [-3000],
            "DAYS_ID_PUBLISH": [-1000],
        }
    )

    original_columns = dataframe.columns.tolist()

    application_feature_engineer.transform(
        dataframe,
    )

    assert dataframe.columns.tolist() == original_columns