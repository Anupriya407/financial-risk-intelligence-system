from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.financial_metrics import FinancialMetrics
from app.repositories.base import BaseRepository


class FinancialMetricsRepository(BaseRepository[FinancialMetrics]):
    """Repository for FinancialMetrics database operations."""

    def __init__(
        self,
        db: Session,
    ) -> None:
        super().__init__(
            db=db,
            model=FinancialMetrics,
        )

    def get_by_financial_statement(
        self,
        financial_statement_id: int,
    ) -> FinancialMetrics | None:
        """Retrieve financial metrics for a financial statement."""

        statement = select(FinancialMetrics).where(
            FinancialMetrics.financial_statement_id == financial_statement_id
        )

        return self.db.scalars(statement).first()

    def get_by_current_ratio(
        self,
        minimum_ratio: float,
    ) -> list[FinancialMetrics]:
        """Retrieve financial metrics with a minimum current ratio."""

        statement = select(FinancialMetrics).where(
            FinancialMetrics.current_ratio >= minimum_ratio
        )

        return list(self.db.scalars(statement).all())

    def get_by_debt_to_equity_ratio(
        self,
        maximum_ratio: float,
    ) -> list[FinancialMetrics]:
        """Retrieve financial metrics below a maximum debt-to-equity ratio."""

        statement = select(FinancialMetrics).where(
            FinancialMetrics.debt_to_equity_ratio <= maximum_ratio
        )

        return list(self.db.scalars(statement).all())

    def get_profitable_companies(
        self,
    ) -> list[FinancialMetrics]:
        """Retrieve financial metrics with a positive net profit margin."""

        statement = select(FinancialMetrics).where(
            FinancialMetrics.net_profit_margin > 0
        )

        return list(self.db.scalars(statement).all())
