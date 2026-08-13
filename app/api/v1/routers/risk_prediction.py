from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas.risk_prediction import (
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from app.dependencies.risk_prediction import (
    get_risk_prediction_service,
)
from app.ml.risk.assessment import RiskAssessmentService


router = APIRouter(
    prefix="/risk-predictions",
    tags=["Risk Predictions"],
)


@router.post(
    "",
    response_model=RiskPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict financial risk",
)
def predict_risk(
    request: RiskPredictionRequest,
    risk_prediction_service: RiskAssessmentService = Depends(
        get_risk_prediction_service,
    ),
) -> RiskPredictionResponse:
    """Generate an ML-based financial risk prediction."""

    try:
        features = pd.DataFrame(
            [request.features]
        )

        prediction = risk_prediction_service.assess(
            features
        )

        return RiskPredictionResponse(
            prediction=prediction
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc