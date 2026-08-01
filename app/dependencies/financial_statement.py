from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.financial_statement_repository import (
    FinancialStatementRepository,
)
from app.services.financial_statement_service import (
    FinancialStatementService,
)


def get_financial_statement_repository(
    db: Session = Depends(get_db),
) -> FinancialStatementRepository:
    """Return a FinancialStatementRepository instance."""

    return FinancialStatementRepository(db)


def get_financial_statement_service(
    financial_statement_repository: FinancialStatementRepository = Depends(
        get_financial_statement_repository,
    ),
) -> FinancialStatementService:
    """Return a FinancialStatementService instance."""

    return FinancialStatementService(
        financial_statement_repository,
    )
