"""
LIFEOS Financial Risk & Stress Testing API

Endpoints for retrieving the five-pillar risk diagnosis,
macroeconomic stress test suite, and executing custom financial stress tests.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User
from app.schemas.schemas import (
    FinancialRiskResponse,
    StressTestScenarioResult,
    CustomStressTestRequest,
)
from app.services.auth_service import get_current_user
from app.services.risk_engine import RiskEngine

router = APIRouter(prefix="/api/risk", tags=["Financial Risk"])


@router.get("/assessment", response_model=FinancialRiskResponse)
async def get_financial_risk_assessment(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Produce comprehensive financial risk diagnostic:
    - Five-pillar risk breakdown (Liquidity, DTI, Cashflow, Obligations, Concentration)
    - Composite risk score (0-100) and risk grade (A/B/C/D/F)
    - Core ratios (DTI, Runway, Savings Rate)
    - Actionable vulnerability warnings & recommendations
    - Standardized stress test matrix (Income loss, rate hikes, emergency shocks, inflation)
    """
    engine = RiskEngine(db, current_user.id)
    return await engine.get_assessment()


@router.post("/stress-test", response_model=StressTestScenarioResult)
async def run_custom_stress_test(
    request: CustomStressTestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Execute a customized multi-variable financial stress test:
    Simulates combined income reduction, unexpected emergency shock,
    interest rate increases, and inflationary pressure over a defined horizon.
    """
    engine = RiskEngine(db, current_user.id)
    return await engine.run_custom_stress_test(
        income_drop_pct=request.income_drop_pct or 0.0,
        emergency_expense=request.emergency_expense,
        interest_rate_hike_bps=request.interest_rate_hike_bps or 0,
        inflation_pct=request.inflation_pct or 0.0,
        duration_months=request.duration_months or 3,
    )
