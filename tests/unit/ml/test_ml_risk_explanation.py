from pathlib import Path

import pandas as pd
import pytest
import joblib

from app.ml.risk.explanation import extract_risk_factors
from app.ml.risk.schemas import RiskFactor


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)

FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


def test_extract_risk_factors_returns_requested_count() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    factors = extract_risk_factors(
        model=model,
        features=features,
        top_n=5,
    )

    assert len(factors) == 5


def test_risk_factors_have_required_fields() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    factors = extract_risk_factors(
        model=model,
        features=features,
        top_n=5,
    )

    for factor in factors:
        assert isinstance(
            factor,
            RiskFactor,
        )

        assert factor.feature

        assert isinstance(
            factor.feature,
            str,
        )

        assert isinstance(
            factor.importance,
            float,
        )


def test_risk_factors_are_ranked_descending() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    factors = extract_risk_factors(
        model=model,
        features=features,
        top_n=5,
    )

    importances = [
        factor.importance
        for factor in factors
    ]

    assert importances == sorted(
        importances,
        reverse=True,
    )


def test_invalid_top_n_is_rejected() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    with pytest.raises(ValueError):
        extract_risk_factors(
            model=model,
            features=features,
            top_n=0,
        )