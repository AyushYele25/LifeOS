"""
LIFEOS Expense Forecast API

Endpoints for retrieving 30/60/90-day hybrid expense projections
combining contractual floor (obligations) with Holt-Winters seasonal baseline,
and evaluating backtest accuracy.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User
from app.schemas.schemas import ForecastResponse, ForecastBacktestResponse
from app.services.auth_service import get_current_user
from app.services.forecast_engine import ForecastEngine

router = APIRouter(prefix="/api/forecast", tags=["Expense Forecast"])


@router.get("", response_model=ForecastResponse)
async def get_expense_forecast(
    as_of: Optional[date] = Query(default=None, description="Evaluation baseline date (defaults to today)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Produce 30, 60, and 90-day expense forecasts.
    Combines deterministic contractual floors with seasonal adjustments
    and quantile confidence intervals (P10, P50, P90).
    """
    engine = ForecastEngine(db, current_user.id)
    return await engine.get_forecast(as_of=as_of)


@router.get("/backtest", response_model=ForecastBacktestResponse)
async def backtest_expense_forecast(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run backtest evaluation against the user's most recent completed calendar month
    to calculate Mean Absolute Percentage Error (MAPE) and overall forecast accuracy.
    """
    engine = ForecastEngine(db, current_user.id)
    return await engine.run_backtest()
