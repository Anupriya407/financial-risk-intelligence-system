from datetime import datetime

from pydantic import Field

from app.api.v1.schemas.base import BaseSchema


class CompanyCreate(BaseSchema):
    """Schema for creating a company."""

    company_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    ticker_symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
    )

    industry: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    country: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class CompanyUpdate(BaseSchema):
    """Schema for updating a company."""

    company_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    ticker_symbol: str | None = Field(
        default=None,
        min_length=1,
        max_length=20,
    )

    industry: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    country: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class CompanyResponse(BaseSchema):
    """Schema returned by the API."""

    id: int
    company_name: str
    ticker_symbol: str
    industry: str
    country: str
    created_at: datetime
    updated_at: datetime
