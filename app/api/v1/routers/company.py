from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas.company import (
    CompanyCreate,
    CompanyResponse,
    CompanyUpdate,
)
from app.dependencies.company import get_company_service
from app.models.company import Company
from app.services.company_service import CompanyService

router = APIRouter(
    prefix="/companies",
    tags=["Companies"],
)


@router.post(
    "",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a company",
    description="Create a new company in the Financial Risk Intelligence System.",
)
def create_company(
    company_data: CompanyCreate,
    company_service: CompanyService = Depends(get_company_service),
) -> CompanyResponse:
    """Create a new company."""

    company = Company(
        company_name=company_data.company_name,
        ticker_symbol=company_data.ticker_symbol,
        industry=company_data.industry,
        country=company_data.country,
    )

    created_company = company_service.create_company(company)

    return CompanyResponse.model_validate(created_company)


@router.get(
    "/{company_id}",
    response_model=CompanyResponse,
    status_code=status.HTTP_200_OK,
    summary="Get company by ID",
    description="Retrieve a company by its unique identifier.",
)
def get_company(
    company_id: int,
    company_service: CompanyService = Depends(get_company_service),
) -> CompanyResponse:
    """Retrieve a company by its ID."""

    company = company_service.get_company_by_id(company_id)

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with ID {company_id} not found.",
        )

    return CompanyResponse.model_validate(company)


@router.get(
    "",
    response_model=list[CompanyResponse],
    status_code=status.HTTP_200_OK,
    summary="Get all companies",
    description="Retrieve all companies in the Financial Risk Intelligence System.",
)
def get_companies(
    company_service: CompanyService = Depends(get_company_service),
) -> list[CompanyResponse]:
    """Retrieve all companies."""

    companies = company_service.get_all_companies()

    return [CompanyResponse.model_validate(company) for company in companies]


@router.put(
    "/{company_id}",
    response_model=CompanyResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a company",
    description="Update an existing company.",
)
def update_company(
    company_id: int,
    company_data: CompanyUpdate,
    company_service: CompanyService = Depends(get_company_service),
) -> CompanyResponse:
    """Update an existing company."""

    company = Company(
        company_name=company_data.company_name,
        ticker_symbol=company_data.ticker_symbol,
        industry=company_data.industry,
        country=company_data.country,
    )

    try:
        updated_company = company_service.update_company(
            company_id,
            company,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return CompanyResponse.model_validate(updated_company)


@router.delete(
    "/{company_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a company",
    description="Delete an existing company from the Financial Risk Intelligence System.",
)
def delete_company(
    company_id: int,
    company_service: CompanyService = Depends(get_company_service),
) -> None:
    """Delete an existing company."""

    try:
        company_service.delete_company(company_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
