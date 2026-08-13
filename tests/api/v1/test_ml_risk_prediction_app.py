from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app


def test_risk_prediction_route_is_registered() -> None:
    paths = app.openapi()["paths"]

    assert "/api/v1/risk-predictions" in paths


def test_risk_prediction_route_is_post() -> None:
    paths = app.openapi()["paths"]

    assert "post" in paths["/api/v1/risk-predictions"]


def test_application_starts_successfully() -> None:
    client = TestClient(app)

    response = client.get("/redoc")

    assert response.status_code == 200