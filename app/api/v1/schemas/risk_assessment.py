from datetime import datetime

from pydantic import Field

from app.api.v1.schemas.base import BaseSchema


class RiskAssessmentCreate(BaseSchema):
    """Schema for creating a risk assessment."""

    company_id: int

    financial_statement_id: int

    risk_score: float

    risk_level: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    model_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )


class RiskAssessmentUpdate(BaseSchema):
    """Schema for updating a risk assessment."""

    company_id: int | None = None

    financial_statement_id: int | None = None

    risk_score: float | None = None

    risk_level: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    model_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )


class RiskAssessmentResponse(BaseSchema):
    """Schema returned by the API."""

    id: int

    company_id: int

    financial_statement_id: int

    risk_score: float

    risk_level: str

    model_version: str

    created_at: datetime

    updated_at: datetime
