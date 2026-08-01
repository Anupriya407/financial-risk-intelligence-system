from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.schemas.risk_assessment import (
    RiskAssessmentCreate,
    RiskAssessmentResponse,
    RiskAssessmentUpdate,
)
from app.dependencies.risk_assessment import (
    get_risk_assessment_service,
)
from app.models.risk_assessment import RiskAssessment
from app.services.risk_assessment_service import (
    RiskAssessmentService,
)

router = APIRouter(
    prefix="/risk-assessments",
    tags=["Risk Assessments"],
)


@router.post(
    "",
    response_model=RiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create risk assessment",
)
def create_risk_assessment(
    risk_assessment_data: RiskAssessmentCreate,
    risk_assessment_service: RiskAssessmentService = Depends(
        get_risk_assessment_service,
    ),
) -> RiskAssessmentResponse:
    """Create a risk assessment."""

    risk_assessment = RiskAssessment(
        company_id=risk_assessment_data.company_id,
        financial_statement_id=(risk_assessment_data.financial_statement_id),
        risk_score=risk_assessment_data.risk_score,
        risk_level=risk_assessment_data.risk_level,
        model_version=risk_assessment_data.model_version,
    )

    created_assessment = risk_assessment_service.create_risk_assessment(risk_assessment)

    return RiskAssessmentResponse.model_validate(created_assessment)


@router.get(
    "/{risk_assessment_id}",
    response_model=RiskAssessmentResponse,
    summary="Get risk assessment by ID",
)
def get_risk_assessment(
    risk_assessment_id: int,
    risk_assessment_service: RiskAssessmentService = Depends(
        get_risk_assessment_service,
    ),
) -> RiskAssessmentResponse:
    """Retrieve a risk assessment."""

    risk_assessment = risk_assessment_service.get_risk_assessment_by_id(risk_assessment_id)

    if risk_assessment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk assessment not found.",
        )

    return RiskAssessmentResponse.model_validate(risk_assessment)


@router.get(
    "",
    response_model=list[RiskAssessmentResponse],
    summary="Get all risk assessments",
)
def get_all_risk_assessments(
    risk_assessment_service: RiskAssessmentService = Depends(
        get_risk_assessment_service,
    ),
) -> list[RiskAssessmentResponse]:
    """Retrieve all risk assessments."""

    assessments = risk_assessment_service.get_all_risk_assessments()

    return [RiskAssessmentResponse.model_validate(item) for item in assessments]


@router.put(
    "/{risk_assessment_id}",
    response_model=RiskAssessmentResponse,
    summary="Update risk assessment",
)
def update_risk_assessment(
    risk_assessment_id: int,
    risk_assessment_data: RiskAssessmentUpdate,
    risk_assessment_service: RiskAssessmentService = Depends(
        get_risk_assessment_service,
    ),
) -> RiskAssessmentResponse:
    """Update a risk assessment."""

    risk_assessment = RiskAssessment(
        company_id=risk_assessment_data.company_id,
        financial_statement_id=(risk_assessment_data.financial_statement_id),
        risk_score=risk_assessment_data.risk_score,
        risk_level=risk_assessment_data.risk_level,
        model_version=risk_assessment_data.model_version,
    )

    try:
        updated_assessment = risk_assessment_service.update_risk_assessment(
            risk_assessment_id,
            risk_assessment,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return RiskAssessmentResponse.model_validate(updated_assessment)


@router.delete(
    "/{risk_assessment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete risk assessment",
)
def delete_risk_assessment(
    risk_assessment_id: int,
    risk_assessment_service: RiskAssessmentService = Depends(
        get_risk_assessment_service,
    ),
) -> None:
    """Delete a risk assessment."""

    try:
        risk_assessment_service.delete_risk_assessment(risk_assessment_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
