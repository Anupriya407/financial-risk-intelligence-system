from fastapi.testclient import TestClient

from app.main import app


def test_real_application_risk_prediction() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/risk-predictions",
        json={
            "features": {
                "SK_ID_CURR": 100001,
            }
        },
    )

    assert response.status_code in {200, 422}