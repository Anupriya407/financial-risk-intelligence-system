from fastapi import APIRouter

from app.api.v1.routers.company import (
    router as company_router,
)
from app.api.v1.routers.financial_metrics import (
    router as financial_metrics_router,
)
from app.api.v1.routers.financial_statement import (
    router as financial_statement_router,
)
from app.api.v1.routers.prediction import (
    router as prediction_router,
)
from app.api.v1.routers.risk_assessment import (
    router as risk_assessment_router,
)

api_router = APIRouter()

api_router.include_router(company_router)
api_router.include_router(financial_statement_router)
api_router.include_router(financial_metrics_router)
api_router.include_router(risk_assessment_router)
api_router.include_router(prediction_router)
