"""
LIFEOS Decision Simulator API

Endpoints for executing and retrieving "Can I afford X?" decision scenarios,
evaluating upfront cash vs EMI financing vs delayed acquisition,
and analyzing new loan and salary adjustments.
"""

from typing import List
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User
from app.schemas.schemas import SimulationRequest, SimulationResponse
from app.services.auth_service import get_current_user
from app.services.scenario_simulator import ScenarioSimulator

router = APIRouter(prefix="/api/simulations", tags=["Decision Simulator"])


@router.post("", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    request: SimulationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run a decision simulation:
    - Compares Cash upfront vs EMI financing vs Saving up & delaying
    - Computes impact on emergency runway, liquid balances, and debt-to-income
    - Provides deterministic mathematical recommendations
    - Persists the result for historical review and comparison
    """
    engine = ScenarioSimulator(db, current_user.id)
    sim = await engine.run_simulation(
        simulation_type=request.simulation_type,
        title=request.title,
        params=request.params,
        persist=True,
    )
    return sim


@router.get("", response_model=List[SimulationResponse])
async def list_simulations(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve historical simulations conducted by the authenticated user.
    """
    engine = ScenarioSimulator(db, current_user.id)
    return await engine.get_history(limit=limit)


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(
    simulation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch a specific simulation by ID with per-user data isolation.
    """
    engine = ScenarioSimulator(db, current_user.id)
    sim = await engine.get_simulation(simulation_id)
    if not sim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found",
        )
    return sim
