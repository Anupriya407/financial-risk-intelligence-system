from pathlib import Path

import pandas as pd

from app.ml.risk.assessment import RiskAssessmentService
from app.ml.risk.schemas import RiskPrediction


MODEL_PATH = Path(
    "data/processed/ml/best_model.joblib"
)

FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


def test_risk_prediction_serializes_to_dict() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(
        features
    )

    data = result.model_dump()

    assert isinstance(
        data,
        dict,
    )

    assert data["model_name"] == "xgboost"
    assert data["model_version"] == "1.0.0"

    assert 0.0 <= data["risk_probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0

    assert data["risk_band"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }


def test_risk_prediction_serializes_to_json() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(
        features
    )

    json_data = result.model_dump_json()

    assert isinstance(
        json_data,
        str,
    )

    assert "risk_probability" in json_data
    assert "risk_score" in json_data
    assert "risk_band" in json_data
    assert "model_name" in json_data
    assert "model_version" in json_data
    assert "risk_factors" in json_data


def test_serialized_risk_factors_are_structured() -> None:
    service = RiskAssessmentService(
        model_path=str(MODEL_PATH)
    )

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    result = service.assess(
        features
    )

    data = result.model_dump()

    assert len(data["risk_factors"]) == 5

    for factor in data["risk_factors"]:
        assert set(factor.keys()) == {
            "feature",
            "importance",
        }

        assert isinstance(
            factor["feature"],
            str,
        )

        assert isinstance(
            factor["importance"],
            float,
        )