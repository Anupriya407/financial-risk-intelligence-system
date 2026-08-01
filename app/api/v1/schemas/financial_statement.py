from datetime import date, datetime

from pydantic import Field

from app.api.v1.schemas.base import BaseSchema


class FinancialStatementCreate(BaseSchema):
    """Schema for creating a financial statement."""

    company_id: int

    reporting_period: date

    fiscal_year: int = Field(
        ...,
        ge=1900,
    )

    statement_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )


class FinancialStatementUpdate(BaseSchema):
    """Schema for updating a financial statement."""

    company_id: int | None = None

    reporting_period: date | None = None

    fiscal_year: int | None = Field(
        default=None,
        ge=1900,
    )

    statement_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )


class FinancialStatementResponse(BaseSchema):
    """Schema returned by the API."""

    id: int

    company_id: int

    reporting_period: date

    fiscal_year: int

    statement_type: str

    created_at: datetime

    updated_at: datetime
