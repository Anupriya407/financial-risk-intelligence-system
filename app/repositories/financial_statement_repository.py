from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models.financial_statement import FinancialStatement
from app.repositories.base import BaseRepository


class FinancialStatementRepository(BaseRepository[FinancialStatement]):
    """Repository for FinancialStatement database operations."""

    def __init__(
        self,
        db: Session,
    ) -> None:
        super().__init__(
            db=db,
            model=FinancialStatement,
        )

    def get_by_company(
        self,
        company_id: int,
    ) -> list[FinancialStatement]:
        """Retrieve all financial statements for a company."""

        statement = select(FinancialStatement).where(
            FinancialStatement.company_id == company_id
        )

        return list(self.db.scalars(statement).all())

    def get_by_fiscal_year(
        self,
        fiscal_year: int,
    ) -> list[FinancialStatement]:
        """Retrieve all financial statements for a fiscal year."""

        statement = select(FinancialStatement).where(
            FinancialStatement.fiscal_year == fiscal_year
        )

        return list(self.db.scalars(statement).all())

    def get_by_statement_type(
        self,
        statement_type: str,
    ) -> list[FinancialStatement]:
        """Retrieve all financial statements by statement type."""

        statement = select(FinancialStatement).where(
            FinancialStatement.statement_type == statement_type
        )

        return list(self.db.scalars(statement).all())

    def get_latest_statement(
        self,
        company_id: int,
    ) -> FinancialStatement | None:
        """Retrieve the latest financial statement for a company."""

        statement = (
            select(FinancialStatement)
            .where(FinancialStatement.company_id == company_id)
            .order_by(desc(FinancialStatement.reporting_period))
        )

        return self.db.scalars(statement).first()
