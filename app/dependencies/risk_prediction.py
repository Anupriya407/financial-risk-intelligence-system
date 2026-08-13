from app.ml.config.training import MLTrainingConfig
from app.ml.risk.assessment import RiskAssessmentService


def get_risk_prediction_service() -> RiskAssessmentService:
    """Return the FRIS ML risk assessment service."""

    config = MLTrainingConfig()

    return RiskAssessmentService(
        model_path=str(
            config.model_artifact_path()
        )
    )