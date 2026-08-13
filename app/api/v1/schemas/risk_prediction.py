from pydantic import BaseModel, Field

from app.ml.risk.schemas import RiskPrediction


class RiskPredictionRequest(BaseModel):
    """Request payload for ML risk prediction."""

    features: dict[str, float | int | str | None] = Field(
        min_length=1,
        description="Customer financial features used by the ML model.",
    )


class RiskPredictionResponse(BaseModel):
    """API response for ML risk prediction."""

    prediction: RiskPrediction