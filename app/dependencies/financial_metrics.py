from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.financial_metrics_repository import (
    FinancialMetricsRepository,
)
from app.services.financial_metrics_service import (
    FinancialMetricsService,
)


def get_financial_metrics_repository(
    db: Session = Depends(get_db),
) -> FinancialMetricsRepository:
    """Return a FinancialMetricsRepository instance."""

    return FinancialMetricsRepository(db)


def get_financial_metrics_service(
    financial_metrics_repository: FinancialMetricsRepository = Depends(
        get_financial_metrics_repository,
    ),
) -> FinancialMetricsService:
    """Return a FinancialMetricsService instance."""

    return FinancialMetricsService(
        financial_metrics_repository,
    )
