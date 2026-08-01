from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas.financial_metrics import (
    FinancialMetricsCreate,
    FinancialMetricsResponse,
    FinancialMetricsUpdate,
)
from app.dependencies.financial_metrics import (
    get_financial_metrics_service,
)
from app.models.financial_metrics import FinancialMetrics
from app.services.financial_metrics_service import (
    FinancialMetricsService,
)

router = APIRouter(
    prefix="/financial-metrics",
    tags=["Financial Metrics"],
)


@router.post(
    "",
    response_model=FinancialMetricsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create financial metrics",
)
def create_financial_metrics(
    financial_metrics_data: FinancialMetricsCreate,
    financial_metrics_service: FinancialMetricsService = Depends(
        get_financial_metrics_service,
    ),
) -> FinancialMetricsResponse:
    """Create financial metrics."""

    financial_metrics = FinancialMetrics(
        financial_statement_id=(financial_metrics_data.financial_statement_id),
        current_ratio=financial_metrics_data.current_ratio,
        debt_to_equity_ratio=(financial_metrics_data.debt_to_equity_ratio),
        return_on_assets=(financial_metrics_data.return_on_assets),
        return_on_equity=(financial_metrics_data.return_on_equity),
        net_profit_margin=(financial_metrics_data.net_profit_margin),
    )

    created_metrics = financial_metrics_service.create_financial_metrics(financial_metrics)

    return FinancialMetricsResponse.model_validate(created_metrics)


@router.get(
    "/{financial_metrics_id}",
    response_model=FinancialMetricsResponse,
    summary="Get financial metrics by ID",
)
def get_financial_metrics(
    financial_metrics_id: int,
    financial_metrics_service: FinancialMetricsService = Depends(
        get_financial_metrics_service,
    ),
) -> FinancialMetricsResponse:
    """Retrieve financial metrics."""

    financial_metrics = financial_metrics_service.get_financial_metrics_by_id(financial_metrics_id)

    if financial_metrics is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Financial metrics not found.",
        )

    return FinancialMetricsResponse.model_validate(financial_metrics)


@router.get(
    "",
    response_model=list[FinancialMetricsResponse],
    summary="Get all financial metrics",
)
def get_all_financial_metrics(
    financial_metrics_service: FinancialMetricsService = Depends(
        get_financial_metrics_service,
    ),
) -> list[FinancialMetricsResponse]:
    """Retrieve all financial metrics."""

    financial_metrics = financial_metrics_service.get_all_financial_metrics()

    return [FinancialMetricsResponse.model_validate(metric) for metric in financial_metrics]


@router.put(
    "/{financial_metrics_id}",
    response_model=FinancialMetricsResponse,
    summary="Update financial metrics",
)
def update_financial_metrics(
    financial_metrics_id: int,
    financial_metrics_data: FinancialMetricsUpdate,
    financial_metrics_service: FinancialMetricsService = Depends(
        get_financial_metrics_service,
    ),
) -> FinancialMetricsResponse:
    """Update financial metrics."""

    financial_metrics = FinancialMetrics(
        financial_statement_id=(financial_metrics_data.financial_statement_id),
        current_ratio=financial_metrics_data.current_ratio,
        debt_to_equity_ratio=(financial_metrics_data.debt_to_equity_ratio),
        return_on_assets=(financial_metrics_data.return_on_assets),
        return_on_equity=(financial_metrics_data.return_on_equity),
        net_profit_margin=(financial_metrics_data.net_profit_margin),
    )

    try:
        updated_metrics = financial_metrics_service.update_financial_metrics(
            financial_metrics_id,
            financial_metrics,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FinancialMetricsResponse.model_validate(updated_metrics)


@router.delete(
    "/{financial_metrics_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete financial metrics",
)
def delete_financial_metrics(
    financial_metrics_id: int,
    financial_metrics_service: FinancialMetricsService = Depends(
        get_financial_metrics_service,
    ),
) -> None:
    """Delete financial metrics."""

    try:
        financial_metrics_service.delete_financial_metrics(financial_metrics_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
