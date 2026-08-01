from datetime import datetime
from typing import Any

from pydantic import Field

from app.api.v1.schemas.base import BaseSchema


class PredictionCreate(BaseSchema):
    """Schema for creating a prediction."""

    risk_assessment_id: int

    predicted_class: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    confidence_score: float

    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    model_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    prediction_metadata: dict[str, Any]


class PredictionUpdate(BaseSchema):
    """Schema for updating a prediction."""

    risk_assessment_id: int | None = None

    predicted_class: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    confidence_score: float | None = None

    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    model_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    prediction_metadata: dict[str, Any] | None = None


class PredictionResponse(BaseSchema):
    """Schema returned by the API."""

    id: int

    risk_assessment_id: int

    predicted_class: str

    confidence_score: float

    model_name: str

    model_version: str

    prediction_metadata: dict[str, Any]

    created_at: datetime

    updated_at: datetime
