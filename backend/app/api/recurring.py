"""
LIFEOS Recurring Payments API Routes.

Endpoints for reviewing, triggering, confirming, and dismissing recurring payment patterns.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from uuid import UUID
from typing import List, Optional
from pydantic import BaseModel
from datetime import date
from decimal import Decimal

from app.database import get_db
from app.models import User, RecurringPattern, AuditLog
from app.models.models import SubscriptionFrequency
from app.services.auth_service import get_current_user
from app.services.recurring_detector import RecurringDetector

router = APIRouter(prefix="/api/recurring-payments", tags=["Recurring Payments"])


class RecurringPatternResponse(BaseModel):
    id: UUID
    merchant: str
    category: Optional[str]
    average_amount: Decimal
    frequency: SubscriptionFrequency
    average_interval_days: Optional[float]
    last_transaction_date: Optional[date]
    next_expected_date: Optional[date]
    confidence: float
    transaction_count: int
    is_confirmed: bool

    model_config = {"from_attributes": True}


@router.get("", response_model=List[RecurringPatternResponse])
async def list_recurring_patterns(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all detected recurring payment patterns for the current user."""
    result = await db.execute(
        select(RecurringPattern)
        .where(RecurringPattern.user_id == current_user.id)
        .order_by(desc(RecurringPattern.confidence))
    )
    return [RecurringPatternResponse.model_validate(p) for p in result.scalars()]


@router.post("/detect", response_model=dict)
async def trigger_detection(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run the recurring payment detection engine on user's transactions."""
    detector = RecurringDetector()
    patterns = await detector.detect_and_save_for_user(current_user.id, db)

    return {
        "status": "success",
        "detected_count": len(patterns),
        "patterns": [RecurringPatternResponse.model_validate(p) for p in patterns],
    }


@router.patch("/{pattern_id}/confirm", response_model=RecurringPatternResponse)
async def confirm_recurring_pattern(
    pattern_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """User confirms a detected candidate pattern as a recurring obligation."""
    result = await db.execute(
        select(RecurringPattern).where(
            and_(
                RecurringPattern.id == pattern_id,
                RecurringPattern.user_id == current_user.id,
            )
        )
    )
    pattern = result.scalar_one_or_none()
    if not pattern:
        raise HTTPException(status_code=404, detail="Recurring pattern not found")

    pattern.is_confirmed = True

    audit = AuditLog(
        user_id=current_user.id,
        action="confirm_recurring_pattern",
        entity_type="recurring_pattern",
        entity_id=str(pattern.id),
        details={"merchant": pattern.merchant, "amount": float(pattern.average_amount)},
    )
    db.add(audit)
    await db.flush()
    await db.refresh(pattern)
    return RecurringPatternResponse.model_validate(pattern)


@router.patch("/{pattern_id}/dismiss", status_code=status.HTTP_204_NO_CONTENT)
async def dismiss_recurring_pattern(
    pattern_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """User dismisses a recurring candidate pattern."""
    result = await db.execute(
        select(RecurringPattern).where(
            and_(
                RecurringPattern.id == pattern_id,
                RecurringPattern.user_id == current_user.id,
            )
        )
    )
    pattern = result.scalar_one_or_none()
    if not pattern:
        raise HTTPException(status_code=404, detail="Recurring pattern not found")

    await db.delete(pattern)
    await db.flush()
