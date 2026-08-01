from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas.financial_statement import (
    FinancialStatementCreate,
    FinancialStatementResponse,
    FinancialStatementUpdate,
)
from app.dependencies.financial_statement import (
    get_financial_statement_service,
)
from app.models.financial_statement import FinancialStatement
from app.services.financial_statement_service import (
    FinancialStatementService,
)

router = APIRouter(
    prefix="/financial-statements",
    tags=["Financial Statements"],
)


@router.post(
    "",
    response_model=FinancialStatementResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a financial statement",
    description="Create a new financial statement.",
)
def create_financial_statement(
    financial_statement_data: FinancialStatementCreate,
    financial_statement_service: FinancialStatementService = Depends(
        get_financial_statement_service,
    ),
) -> FinancialStatementResponse:
    """Create a financial statement."""

    financial_statement = FinancialStatement(
        company_id=financial_statement_data.company_id,
        reporting_period=financial_statement_data.reporting_period,
        fiscal_year=financial_statement_data.fiscal_year,
        statement_type=financial_statement_data.statement_type,
    )

    created_statement = financial_statement_service.create_financial_statement(financial_statement)

    return FinancialStatementResponse.model_validate(created_statement)


@router.get(
    "/{financial_statement_id}",
    response_model=FinancialStatementResponse,
    status_code=status.HTTP_200_OK,
    summary="Get financial statement by ID",
    description="Retrieve a financial statement by its ID.",
)
def get_financial_statement(
    financial_statement_id: int,
    financial_statement_service: FinancialStatementService = Depends(
        get_financial_statement_service,
    ),
) -> FinancialStatementResponse:
    """Retrieve a financial statement."""

    financial_statement = financial_statement_service.get_financial_statement_by_id(
        financial_statement_id,
    )

    if financial_statement is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Financial statement with ID {financial_statement_id} not found."),
        )

    return FinancialStatementResponse.model_validate(financial_statement)


@router.get(
    "",
    response_model=list[FinancialStatementResponse],
    status_code=status.HTTP_200_OK,
    summary="Get all financial statements",
    description="Retrieve all financial statements.",
)
def get_financial_statements(
    financial_statement_service: FinancialStatementService = Depends(
        get_financial_statement_service,
    ),
) -> list[FinancialStatementResponse]:
    """Retrieve all financial statements."""

    financial_statements = financial_statement_service.get_all_financial_statements()

    return [FinancialStatementResponse.model_validate(statement) for statement in financial_statements]


@router.put(
    "/{financial_statement_id}",
    response_model=FinancialStatementResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a financial statement",
    description="Update an existing financial statement.",
)
def update_financial_statement(
    financial_statement_id: int,
    financial_statement_data: FinancialStatementUpdate,
    financial_statement_service: FinancialStatementService = Depends(
        get_financial_statement_service,
    ),
) -> FinancialStatementResponse:
    """Update a financial statement."""

    financial_statement = FinancialStatement(
        company_id=financial_statement_data.company_id,
        reporting_period=financial_statement_data.reporting_period,
        fiscal_year=financial_statement_data.fiscal_year,
        statement_type=financial_statement_data.statement_type,
    )

    try:
        updated_statement = financial_statement_service.update_financial_statement(
            financial_statement_id,
            financial_statement,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FinancialStatementResponse.model_validate(updated_statement)


@router.delete(
    "/{financial_statement_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a financial statement",
    description="Delete an existing financial statement.",
)
def delete_financial_statement(
    financial_statement_id: int,
    financial_statement_service: FinancialStatementService = Depends(
        get_financial_statement_service,
    ),
) -> None:
    """Delete a financial statement."""

    try:
        financial_statement_service.delete_financial_statement(
            financial_statement_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
