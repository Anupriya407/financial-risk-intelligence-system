from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.company import Company
from app.repositories.base import BaseRepository


class CompanyRepository(BaseRepository[Company]):
    """Repository for Company database operations."""

    def __init__(
        self,
        db: Session,
    ) -> None:
        super().__init__(
            db=db,
            model=Company,
        )

    def get_by_ticker(
        self,
        ticker_symbol: str,
    ) -> Company | None:
        """Retrieve a company by its ticker symbol."""

        statement = select(Company).where(Company.ticker_symbol == ticker_symbol)

        return self.db.scalars(statement).first()

    def get_by_company_name(
        self,
        company_name: str,
    ) -> Company | None:
        """Retrieve a company by its name."""

        statement = select(Company).where(Company.company_name == company_name)

        return self.db.scalars(statement).first()

    def get_by_industry(
        self,
        industry: str,
    ) -> list[Company]:
        """Retrieve all companies in a specific industry."""

        statement = select(Company).where(Company.industry == industry)

        return list(self.db.scalars(statement).all())

    def get_by_country(
        self,
        country: str,
    ) -> list[Company]:
        """Retrieve all companies from a specific country."""

        statement = select(Company).where(Company.country == country)

        return list(self.db.scalars(statement).all())
