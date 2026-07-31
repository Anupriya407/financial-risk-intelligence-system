from app.models.financial_statement import FinancialStatement
from app.repositories.financial_statement_repository import (
    FinancialStatementRepository,
)


class FinancialStatementService:
    """Service layer for FinancialStatement business logic."""

    def __init__(
        self,
        financial_statement_repository: FinancialStatementRepository,
    ) -> None:
        self.financial_statement_repository = (
            financial_statement_repository
        )

    def create_financial_statement(
        self,
        financial_statement: FinancialStatement,
    ) -> FinancialStatement:
        """Create a financial statement."""

        return self.financial_statement_repository.create(
            financial_statement
        )

    def get_financial_statement_by_id(
        self,
        financial_statement_id: int,
    ) -> FinancialStatement | None:
        """Retrieve a financial statement by its ID."""

        return self.financial_statement_repository.get_by_id(
            financial_statement_id
        )

    def get_financial_statements_by_company(
        self,
        company_id: int,
    ) -> list[FinancialStatement]:
        """Retrieve all financial statements for a company."""

        return self.financial_statement_repository.get_by_company(
            company_id
        )

    def get_financial_statements_by_fiscal_year(
        self,
        fiscal_year: int,
    ) -> list[FinancialStatement]:
        """Retrieve all financial statements for a fiscal year."""

        return (
            self.financial_statement_repository.get_by_fiscal_year(
                fiscal_year
            )
        )

    def get_financial_statements_by_statement_type(
        self,
        statement_type: str,
    ) -> list[FinancialStatement]:
        """Retrieve financial statements by type."""

        return (
            self.financial_statement_repository.get_by_statement_type(
                statement_type
            )
        )

    def get_latest_financial_statement(
        self,
        company_id: int,
    ) -> FinancialStatement | None:
        """Retrieve the latest financial statement."""

        return (
            self.financial_statement_repository.get_latest_statement(
                company_id
            )
        )

    def update_financial_statement(
        self,
        financial_statement: FinancialStatement,
    ) -> FinancialStatement:
        """Update a financial statement."""

        existing_statement = (
            self.financial_statement_repository.get_by_id(
                financial_statement.id
            )
        )

        if existing_statement is None:
            raise ValueError("Financial statement not found.")

        return self.financial_statement_repository.update(
            financial_statement
        )

    def delete_financial_statement(
        self,
        financial_statement_id: int,
    ) -> None:
        """Delete a financial statement."""

        financial_statement = (
            self.financial_statement_repository.get_by_id(
                financial_statement_id
            )
        )

        if financial_statement is None:
            raise ValueError("Financial statement not found.")

        self.financial_statement_repository.delete(
            financial_statement
        )