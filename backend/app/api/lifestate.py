"""
LIFEOS Life State API

Endpoints for querying, recomputing, and comparing the user's financial life state.
"""

from typing import List, Optional, Dict, Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User, LifeState
from app.schemas.schemas import LifeStateResponse
from app.services.auth_service import get_current_user
from app.services.life_state_engine import LifeStateEngine

router = APIRouter(prefix="/api/life-state", tags=["Life State"])


@router.get("/current", response_model=LifeStateResponse)
async def get_current_life_state(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the user's current Life State snapshot.
    If no snapshot exists, recomputes deterministically on the fly.
    """
    engine = LifeStateEngine(db, current_user.id)
    state = await engine.get_current_state()
    return state


@router.post("/recompute", response_model=LifeStateResponse)
async def recompute_life_state(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Force an immediate recomputation of the user's financial state across
    all accounts, transactions, loans, insurance, and upcoming obligations.
    Persists a new snapshot in the life_states table.
    """
    engine = LifeStateEngine(db, current_user.id)
    state = await engine.recompute_state(save_snapshot=True)
    return state


@router.get("/history", response_model=List[LifeStateResponse])
async def get_life_state_history(
    limit: int = Query(default=12, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve historical Life State snapshots to monitor financial evolution over time.
    """
    engine = LifeStateEngine(db, current_user.id)
    history = await engine.get_history(limit=limit)
    return history


@router.get("/compare", response_model=Dict[str, Any])
async def compare_life_states(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Compares the most recent snapshot against the previous snapshot.
    Returns delta metrics, percentage shifts, and identified changes.
    """
    engine = LifeStateEngine(db, current_user.id)
    history = await engine.get_history(limit=2)

    if len(history) < 2:
        latest = history[0] if history else await engine.get_current_state()
        return {
            "has_previous": False,
            "current_state": LifeStateResponse.model_validate(latest),
            "previous_state": None,
            "changes": [],
            "summary": "Baseline state created. Further snapshots will show comparative deltas.",
        }

    latest, previous = history[0], history[1]
    deltas = engine._compute_state_deltas(
        previous,
        {
            "total_balance": latest.total_balance,
            "monthly_expenses": latest.monthly_expenses,
            "monthly_income": latest.monthly_income,
            "emergency_runway_months": latest.emergency_runway_months,
            "financial_risk_score": latest.financial_risk_score,
            "upcoming_obligations_30d": latest.upcoming_obligations_30d,
        },
    )

    return {
        "has_previous": True,
        "current_state": LifeStateResponse.model_validate(latest),
        "previous_state": LifeStateResponse.model_validate(previous),
        "changes": deltas,
        "summary": f"Detected {len(deltas)} significant financial shifts between snapshots.",
    }
