from pathlib import Path

import pandas as pd
import pytest

from app.ml.risk.assessment import RiskAssessmentService


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)

FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


def test_missing_model_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError):
        RiskAssessmentService(
            model_path=str(missing_model)
        )


def test_empty_features_are_rejected() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.DataFrame()

    with pytest.raises((ValueError, KeyError)):
        service.assess(features)


def test_multiple_rows_are_rejected() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:2]

    with pytest.raises(ValueError):
        service.assess(features)


def test_prediction_requires_expected_features() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1].copy()

    features["unexpected_feature"] = 123.0

    with pytest.raises((ValueError, KeyError)):
        service.assess(features)