from app.models.company import Company
from app.repositories.company_repository import CompanyRepository


class CompanyService:
    """Service layer for Company business logic."""

    def __init__(
        self,
        company_repository: CompanyRepository,
    ) -> None:
        self.company_repository = company_repository

    def create_company(
        self,
        company: Company,
    ) -> Company:
        """Create a new company."""

        existing_company = self.company_repository.get_by_ticker(
            company.ticker_symbol,
        )

        if existing_company is not None:
            raise ValueError(
                f"Company with ticker '{company.ticker_symbol}' already exists."
            )

        return self.company_repository.create(company)

    def get_company_by_id(
        self,
        company_id: int,
    ) -> Company | None:
        """Retrieve a company by its ID."""

        return self.company_repository.get_by_id(company_id)

    def get_company_by_ticker(
        self,
        ticker_symbol: str,
    ) -> Company | None:
        """Retrieve a company by its ticker."""

        return self.company_repository.get_by_ticker(ticker_symbol)

    def get_company_by_name(
        self,
        company_name: str,
    ) -> Company | None:
        """Retrieve a company by its name."""

        return self.company_repository.get_by_company_name(company_name)

    def get_companies_by_industry(
        self,
        industry: str,
    ) -> list[Company]:
        """Retrieve companies in an industry."""

        return self.company_repository.get_by_industry(industry)

    def get_companies_by_country(
        self,
        country: str,
    ) -> list[Company]:
        """Retrieve companies from a country."""

        return self.company_repository.get_by_country(country)

    def update_company(
        self,
        company: Company,
    ) -> Company:
        """Update a company."""

        existing_company = self.company_repository.get_by_id(company.id)

        if existing_company is None:
            raise ValueError("Company not found.")

        return self.company_repository.update(company)

    def delete_company(
        self,
        company_id: int,
    ) -> None:
        """Delete a company."""

        company = self.company_repository.get_by_id(company_id)

        if company is None:
            raise ValueError("Company not found.")

        self.company_repository.delete(company)