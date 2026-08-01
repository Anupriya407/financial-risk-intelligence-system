from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.company_repository import CompanyRepository
from app.services.company_service import CompanyService


def get_company_repository(
    db: Session = Depends(get_db),
) -> CompanyRepository:
    """Return a CompanyRepository instance."""

    return CompanyRepository(db)


def get_company_service(
    repository: CompanyRepository = Depends(get_company_repository),
) -> CompanyService:
    """Return a CompanyService instance."""

    return CompanyService(repository)
