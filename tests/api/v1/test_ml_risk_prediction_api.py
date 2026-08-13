from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


FEATURES_PATH = Path(
    "data/processed/features/fris_training_features.parquet"
)


client = TestClient(app)


def _build_real_feature_payload() -> dict[str, dict[str, object]]:
    """Build an API payload from one real FRIS feature row."""

    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1].copy()

    features = features.fillna(0)

    row = features.iloc[0]

    payload: dict[str, object] = {}

    for column in features.columns:
        value = row[column]

        if hasattr(value, "item"):
            value = value.item()

        payload[column] = value

    return {
        "features": payload
    }


def test_risk_prediction_endpoint() -> None:
    """Test the complete production risk prediction endpoint."""

    payload = _build_real_feature_payload()

    response = client.post(
        "/api/v1/risk-predictions",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    prediction = data["prediction"]

    assert 0.0 <= prediction["risk_probability"] <= 1.0

    assert 0.0 <= prediction["risk_score"] <= 100.0

    assert prediction["risk_band"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    assert prediction["model_name"] == "xgboost"

    assert prediction["model_version"] == "1.0.0"

    assert "risk_factors" in prediction


def test_risk_prediction_rejects_empty_features() -> None:
    """Empty feature dictionaries must be rejected."""

    response = client.post(
        "/api/v1/risk-predictions",
        json={
            "features": {}
        },
    )

    assert response.status_code == 422


def test_risk_prediction_requires_features() -> None:
    """The features field is mandatory."""

    response = client.post(
        "/api/v1/risk-predictions",
        json={},
    )

    assert response.status_code == 422