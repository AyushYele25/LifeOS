"""
LIFEOS Events API Routes

Forward-looking obligation horizon and event tracking.
Enforces multi-tenant boundary and deterministic calculations.
"""

from datetime import date, timedelta
from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User, Event, EventStatus, EventType, AuditLog
from app.schemas.schemas import (
    EventCreate,
    EventResponse,
    EventStatusUpdate,
    HorizonSummaryResponse,
    EventSyncResult,
)
from app.services.auth_service import get_current_user
from app.services.event_engine import EventEngine

router = APIRouter(prefix="/api/events", tags=["Events"])


@router.get("", response_model=List[EventResponse])
async def list_events(
    days_ahead: int = Query(default=30, ge=1, le=365),
    include_past: bool = Query(default=False),
    event_type: Optional[EventType] = Query(None),
    status: Optional[EventStatus] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get upcoming events/obligations with optional type and status filtering.
    """
    today = date.today()
    query = select(Event).where(Event.user_id == current_user.id)

    if not include_past:
        query = query.where(Event.event_date >= today)
    query = query.where(Event.event_date <= today + timedelta(days=days_ahead))

    if event_type is not None:
        query = query.where(Event.event_type == event_type)
    if status is not None:
        query = query.where(Event.status == status)

    query = query.order_by(Event.event_date.asc())
    result = await db.execute(query)
    events = result.scalars().all()

    # Populate days_until
    for e in events:
        e.days_until = (e.event_date - today).days

    return events


@router.post("", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    data: EventCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a custom future obligation event.
    """
    event = Event(user_id=current_user.id, **data.model_dump())
    db.add(event)
    await db.commit()
    await db.refresh(event)
    event.days_until = (event.event_date - date.today()).days
    return event


@router.post("/sync", response_model=EventSyncResult)
async def sync_obligations(
    horizon_days: int = Query(default=90, ge=30, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Scan active loans, insurance policies, subscriptions, and assets to compile
    a forward-looking obligation horizon and mark overdue events.
    """
    engine = EventEngine(db, current_user.id)
    result = await engine.sync_all_obligations(horizon_days=horizon_days)

    return EventSyncResult(
        scanned_sources=result["scanned_sources"],
        new_events_created=result["new_events_created"],
        overdue_events_marked=result["overdue_events_marked"],
        events=result["events"],
    )


@router.get("/horizon", response_model=HorizonSummaryResponse)
async def get_event_horizon(
    horizon_days: int = Query(default=90, ge=30, le=180),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get exact deterministic 30/60/90-day cash outflow requirement summary,
    grouped by month, category, and chronological timeline.
    """
    engine = EventEngine(db, current_user.id)
    summary = await engine.get_horizon_summary(horizon_days=horizon_days)
    return summary


@router.patch("/{event_id}/complete", response_model=EventResponse)
async def complete_event(
    event_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an event as COMPLETED. If recurring, automatically rolls forward
    to generate the next scheduled occurrence.
    """
    engine = EventEngine(db, current_user.id)
    event = await engine.complete_event(event_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found",
        )
    event.days_until = (event.event_date - date.today()).days
    return event


@router.patch("/{event_id}/status", response_model=EventResponse)
async def update_event_status(
    event_id: uuid.UUID,
    payload: EventStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update event status (UPCOMING, DUE, COMPLETED, OVERDUE, CANCELLED).
    """
    result = await db.execute(
        select(Event).where(
            and_(Event.id == event_id, Event.user_id == current_user.id)
        )
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found",
        )

    event.status = payload.status
    await db.commit()
    await db.refresh(event)
    event.days_until = (event.event_date - date.today()).days
    return event


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    event_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a future obligation event.
    """
    result = await db.execute(
        select(Event).where(
            and_(Event.id == event_id, Event.user_id == current_user.id)
        )
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found",
        )

    await db.delete(event)
    await db.commit()
    return None
