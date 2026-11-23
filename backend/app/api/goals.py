"""
LIFEOS Goals API Routes

CRUD for financial goals and milestones.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import UUID

from app.database import get_db
from app.models import User, Goal
from app.models.models import GoalStatus
from app.schemas.schemas import GoalCreate, GoalUpdate, GoalResponse
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/goals", tags=["Goals"])


@router.get("", response_model=list[GoalResponse])
async def list_goals(
    status: Optional[GoalStatus] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all financial goals for current user."""
    query = select(Goal).where(Goal.user_id == current_user.id)
    if status is not None:
        query = query.where(Goal.status == status)
    result = await db.execute(query.order_by(Goal.priority))
    return [GoalResponse.model_validate(g) for g in result.scalars()]


@router.post("", response_model=GoalResponse, status_code=status.HTTP_201_CREATED)
async def create_goal(
    data: GoalCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new financial goal."""
    goal = Goal(user_id=current_user.id, **data.model_dump())
    db.add(goal)
    await db.flush()
    await db.refresh(goal)
    return GoalResponse.model_validate(goal)


@router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal(
    goal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get single goal details."""
    result = await db.execute(
        select(Goal).where(and_(Goal.id == goal_id, Goal.user_id == current_user.id))
    )
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.patch("/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: UUID,
    data: GoalUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update progress or metadata for a financial goal."""
    result = await db.execute(
        select(Goal).where(and_(Goal.id == goal_id, Goal.user_id == current_user.id))
    )
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(goal, field, value)

    # Check if target reached
    if goal.current_amount >= goal.target_amount and goal.status == GoalStatus.ACTIVE:
        goal.status = GoalStatus.COMPLETED

    await db.flush()
    await db.refresh(goal)
    return GoalResponse.model_validate(goal)


@router.delete("/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_goal(
    goal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a goal."""
    result = await db.execute(
        select(Goal).where(and_(Goal.id == goal_id, Goal.user_id == current_user.id))
    )
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    await db.delete(goal)
    await db.flush()
