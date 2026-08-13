from pathlib import Path

from app.dependencies.risk_prediction import (
    get_risk_prediction_service,
)
from app.ml.config.training import MLTrainingConfig
from app.ml.risk.assessment import RiskAssessmentService


def test_risk_prediction_dependency_returns_service() -> None:
    service = get_risk_prediction_service()

    assert isinstance(
        service,
        RiskAssessmentService,
    )


def test_risk_prediction_dependency_uses_configured_model() -> None:
    service = get_risk_prediction_service()

    config = MLTrainingConfig()

    expected_path = Path(
        config.model_artifact_path()
    )

    actual_path = service.predictor.model_path

    assert actual_path == expected_path


def test_configured_model_artifact_exists() -> None:
    config = MLTrainingConfig()

    model_path = config.model_artifact_path()

    assert model_path.exists()
    assert model_path.is_file()