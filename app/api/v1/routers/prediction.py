from fastapi import APIRouter, Depends, status

from app.api.v1.schemas.prediction import (
    PredictionCreate,
    PredictionResponse,
    PredictionUpdate,
)
from app.dependencies.prediction import (
    get_prediction_service,
)
from app.models.prediction import Prediction
from app.services.prediction_service import (
    PredictionService,
)

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create prediction",
)
def create_prediction(
    prediction_data: PredictionCreate,
    prediction_service: PredictionService = Depends(
        get_prediction_service,
    ),
) -> PredictionResponse:
    """Create a prediction."""

    prediction = Prediction(
        risk_assessment_id=prediction_data.risk_assessment_id,
        predicted_class=prediction_data.predicted_class,
        confidence_score=prediction_data.confidence_score,
        model_name=prediction_data.model_name,
        model_version=prediction_data.model_version,
        prediction_metadata=prediction_data.prediction_metadata,
    )

    created_prediction = prediction_service.create_prediction(
        prediction,
    )

    return PredictionResponse.model_validate(created_prediction)


@router.get(
    "/{prediction_id}",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prediction by ID",
)
def get_prediction(
    prediction_id: int,
    prediction_service: PredictionService = Depends(
        get_prediction_service,
    ),
) -> PredictionResponse:
    """Retrieve a prediction."""

    prediction = prediction_service.get_prediction_by_id(
        prediction_id,
    )

    return PredictionResponse.model_validate(prediction)


@router.get(
    "",
    response_model=list[PredictionResponse],
    status_code=status.HTTP_200_OK,
    summary="Get all predictions",
)
def get_predictions(
    prediction_service: PredictionService = Depends(
        get_prediction_service,
    ),
) -> list[PredictionResponse]:
    """Retrieve all predictions."""

    predictions = prediction_service.get_all_predictions()

    return [
        PredictionResponse.model_validate(prediction)
        for prediction in predictions
    ]


@router.put(
    "/{prediction_id}",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update prediction",
)
def update_prediction(
    prediction_id: int,
    prediction_data: PredictionUpdate,
    prediction_service: PredictionService = Depends(
        get_prediction_service,
    ),
) -> PredictionResponse:
    """Update a prediction."""

    prediction = Prediction(
        risk_assessment_id=prediction_data.risk_assessment_id,
        predicted_class=prediction_data.predicted_class,
        confidence_score=prediction_data.confidence_score,
        model_name=prediction_data.model_name,
        model_version=prediction_data.model_version,
        prediction_metadata=prediction_data.prediction_metadata,
    )

    updated_prediction = prediction_service.update_prediction(
        prediction_id,
        prediction,
    )

    return PredictionResponse.model_validate(updated_prediction)


@router.delete(
    "/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete prediction",
)
def delete_prediction(
    prediction_id: int,
    prediction_service: PredictionService = Depends(
        get_prediction_service,
    ),
) -> None:
    """Delete a prediction."""

    prediction_service.delete_prediction(
        prediction_id,
    )