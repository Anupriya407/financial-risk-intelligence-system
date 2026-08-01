from datetime import datetime

from app.api.v1.schemas.base import BaseSchema


class FinancialMetricsCreate(BaseSchema):
    """Schema for creating financial metrics."""

    financial_statement_id: int

    current_ratio: float

    debt_to_equity_ratio: float

    return_on_assets: float

    return_on_equity: float

    net_profit_margin: float


class FinancialMetricsUpdate(BaseSchema):
    """Schema for updating financial metrics."""

    financial_statement_id: int | None = None

    current_ratio: float | None = None

    debt_to_equity_ratio: float | None = None

    return_on_assets: float | None = None

    return_on_equity: float | None = None

    net_profit_margin: float | None = None


class FinancialMetricsResponse(BaseSchema):
    """Schema returned by the API."""

    id: int

    financial_statement_id: int

    current_ratio: float

    debt_to_equity_ratio: float

    return_on_assets: float

    return_on_equity: float

    net_profit_margin: float

    created_at: datetime

    updated_at: datetime
