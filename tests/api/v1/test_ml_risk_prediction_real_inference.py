import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


FEATURES_PATH = (
    "data/processed/features/fris_training_features.parquet"
)


def test_real_risk_prediction_returns_prediction() -> None:
    features = pd.read_parquet(
        FEATURES_PATH
    ).iloc[:1]

    payload = {
        "features": {
            column: (
                None
                if pd.isna(value)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for column, value in features.iloc[0].items()
        }
    }

    client = TestClient(app)

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

    assert isinstance(
        prediction["risk_factors"],
        list,
    )