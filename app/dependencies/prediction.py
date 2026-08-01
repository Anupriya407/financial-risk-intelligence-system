from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.prediction_repository import (
    PredictionRepository,
)
from app.services.prediction_service import (
    PredictionService,
)


def get_prediction_repository(
    db: Session = Depends(get_db),
) -> PredictionRepository:
    """Return a PredictionRepository instance."""

    return PredictionRepository(db)


def get_prediction_service(
    prediction_repository: PredictionRepository = Depends(
        get_prediction_repository,
    ),
) -> PredictionService:
    """Return a PredictionService instance."""

    return PredictionService(
        prediction_repository,
    )
