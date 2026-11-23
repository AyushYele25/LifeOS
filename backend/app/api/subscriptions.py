"""
LIFEOS Subscriptions API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import UUID

from app.database import get_db
from app.models import User, Subscription
from app.models.models import SubscriptionStatus
from app.schemas.schemas import SubscriptionCreate, SubscriptionResponse
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/subscriptions", tags=["Subscriptions"])


@router.get("", response_model=list[SubscriptionResponse])
async def list_subscriptions(
    include_cancelled: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Subscription).where(Subscription.user_id == current_user.id)
    if not include_cancelled:
        query = query.where(Subscription.status == SubscriptionStatus.ACTIVE)
    result = await db.execute(query)
    return [SubscriptionResponse.model_validate(s) for s in result.scalars()]


@router.post("", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_subscription(
    data: SubscriptionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sub = Subscription(user_id=current_user.id, **data.model_dump())
    db.add(sub)
    await db.flush()
    await db.refresh(sub)
    return SubscriptionResponse.model_validate(sub)


@router.patch("/{subscription_id}/cancel", response_model=SubscriptionResponse)
async def cancel_subscription(
    subscription_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Subscription).where(
            and_(Subscription.id == subscription_id, Subscription.user_id == current_user.id)
        )
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    sub.status = SubscriptionStatus.CANCELLED
    await db.flush()
    await db.refresh(sub)
    return SubscriptionResponse.model_validate(sub)
