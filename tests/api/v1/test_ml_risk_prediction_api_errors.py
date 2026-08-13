from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_missing_features_returns_422() -> None:
    response = client.post(
        "/api/v1/risk-predictions",
        json={},
    )

    assert response.status_code == 422


def test_empty_features_returns_422() -> None:
    response = client.post(
        "/api/v1/risk-predictions",
        json={
            "features": {}
        },
    )

    assert response.status_code == 422


def test_unknown_feature_returns_422() -> None:
    response = client.post(
        "/api/v1/risk-predictions",
        json={
            "features": {
                "unknown_feature": 123.0,
            }
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert "error" in data
    assert data["error"]["code"] == "HTTP_ERROR"
    assert "message" in data["error"]