from app.models.financial_metrics import FinancialMetrics
from app.repositories.financial_metrics_repository import (
    FinancialMetricsRepository,
)


class FinancialMetricsService:
    """Service layer for FinancialMetrics business logic."""

    def __init__(
        self,
        financial_metrics_repository: FinancialMetricsRepository,
    ) -> None:
        self.financial_metrics_repository = financial_metrics_repository

    def create_financial_metrics(
        self,
        financial_metrics: FinancialMetrics,
    ) -> FinancialMetrics:
        """Create financial metrics."""

        return self.financial_metrics_repository.create(financial_metrics)

    def get_financial_metrics_by_id(
        self,
        financial_metrics_id: int,
    ) -> FinancialMetrics | None:
        """Retrieve financial metrics by ID."""

        return self.financial_metrics_repository.get_by_id(financial_metrics_id)

    def get_all_financial_metrics(
        self,
    ) -> list[FinancialMetrics]:
        """Retrieve all financial metrics."""

        return self.financial_metrics_repository.get_all()

    def get_financial_metrics_by_statement(
        self,
        financial_statement_id: int,
    ) -> FinancialMetrics | None:
        """Retrieve financial metrics for a financial statement."""

        return self.financial_metrics_repository.get_by_financial_statement(financial_statement_id)

    def get_financial_metrics_by_current_ratio(
        self,
        minimum_ratio: float,
    ) -> list[FinancialMetrics]:
        """Retrieve financial metrics with a minimum current ratio."""

        return self.financial_metrics_repository.get_by_current_ratio(minimum_ratio)

    def get_financial_metrics_by_debt_to_equity_ratio(
        self,
        maximum_ratio: float,
    ) -> list[FinancialMetrics]:
        """Retrieve financial metrics below a maximum debt-to-equity ratio."""

        return self.financial_metrics_repository.get_by_debt_to_equity_ratio(maximum_ratio)

    def get_profitable_companies(
        self,
    ) -> list[FinancialMetrics]:
        """Retrieve profitable companies."""

        return self.financial_metrics_repository.get_profitable_companies()

    def update_financial_metrics(
        self,
        financial_metrics_id: int,
        financial_metrics_data: FinancialMetrics,
    ) -> FinancialMetrics:
        """Update financial metrics."""

        financial_metrics = self.financial_metrics_repository.get_by_id(financial_metrics_id)

        if financial_metrics is None:
            raise ValueError("Financial metrics not found.")

        financial_metrics.financial_statement_id = financial_metrics_data.financial_statement_id
        financial_metrics.current_ratio = financial_metrics_data.current_ratio
        financial_metrics.debt_to_equity_ratio = financial_metrics_data.debt_to_equity_ratio
        financial_metrics.return_on_assets = financial_metrics_data.return_on_assets
        financial_metrics.return_on_equity = financial_metrics_data.return_on_equity
        financial_metrics.net_profit_margin = financial_metrics_data.net_profit_margin

        return self.financial_metrics_repository.update(financial_metrics)

    def delete_financial_metrics(
        self,
        financial_metrics_id: int,
    ) -> None:
        """Delete financial metrics."""

        financial_metrics = self.financial_metrics_repository.get_by_id(financial_metrics_id)

        if financial_metrics is None:
            raise ValueError("Financial metrics not found.")

        self.financial_metrics_repository.delete(financial_metrics)
