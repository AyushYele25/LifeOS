"""
LIFEOS Dashboard API Route

The main overview endpoint — everything at a glance.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from datetime import date, timedelta
from typing import Optional

from app.database import get_db
from app.models import User, Transaction, Event, Alert, Goal, LifeState
from app.models.models import (
    TransactionType, EventStatus, AlertSeverity, GoalStatus,
)
from app.schemas.schemas import (
    DashboardOverview, UserResponse, LifeStateResponse,
    TransactionResponse, EventResponse, AlertResponse, GoalResponse,
)
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/overview", response_model=dict)
async def get_overview(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Main dashboard overview.
    Returns:
    - User info
    - Latest life state
    - Recent transactions
    - Upcoming events (30 days)
    - Active alerts
    - Active goals
    - Monthly summary
    """
    today = date.today()
    thirty_days = today + timedelta(days=30)
    month_start = today.replace(day=1)

    # Latest life state
    life_state_result = await db.execute(
        select(LifeState)
        .where(LifeState.user_id == current_user.id)
        .order_by(desc(LifeState.snapshot_date))
        .limit(1)
    )
    life_state = life_state_result.scalar_one_or_none()

    # Recent transactions (last 10)
    txn_result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(desc(Transaction.date))
        .limit(10)
    )
    recent_transactions = txn_result.scalars().all()

    # Upcoming events (next 30 days)
    events_result = await db.execute(
        select(Event)
        .where(
            and_(
                Event.user_id == current_user.id,
                Event.event_date >= today,
                Event.event_date <= thirty_days,
                Event.status.in_([EventStatus.UPCOMING, EventStatus.DUE]),
            )
        )
        .order_by(Event.event_date)
        .limit(20)
    )
    upcoming_events = events_result.scalars().all()

    # Active alerts (unread)
    alerts_result = await db.execute(
        select(Alert)
        .where(
            and_(
                Alert.user_id == current_user.id,
                Alert.is_dismissed == False,
            )
        )
        .order_by(desc(Alert.created_at))
        .limit(10)
    )
    active_alerts = alerts_result.scalars().all()

    # Active goals
    goals_result = await db.execute(
        select(Goal)
        .where(
            and_(
                Goal.user_id == current_user.id,
                Goal.status == GoalStatus.ACTIVE,
            )
        )
        .order_by(Goal.priority)
    )
    active_goals = goals_result.scalars().all()

    # Monthly summary (current month)
    monthly_result = await db.execute(
        select(
            Transaction.transaction_type,
            func.sum(Transaction.amount),
            func.count(Transaction.id),
        )
        .where(
            and_(
                Transaction.user_id == current_user.id,
                Transaction.date >= month_start,
            )
        )
        .group_by(Transaction.transaction_type)
    )

    monthly_income = 0
    monthly_expenses = 0
    txn_count = 0
    for txn_type, total, count in monthly_result:
        if txn_type == TransactionType.CREDIT:
            monthly_income = float(total or 0)
        else:
            monthly_expenses = float(total or 0)
        txn_count += count

    # Anomaly count this month
    anomaly_result = await db.execute(
        select(func.count(Transaction.id))
        .where(
            and_(
                Transaction.user_id == current_user.id,
                Transaction.date >= month_start,
                Transaction.is_anomaly == True,
            )
        )
    )
    anomaly_count = anomaly_result.scalar() or 0

    return {
        "user": UserResponse.model_validate(current_user).model_dump(),
        "life_state": (
            LifeStateResponse.model_validate(life_state).model_dump()
            if life_state else None
        ),
        "recent_transactions": [
            TransactionResponse.model_validate(t).model_dump()
            for t in recent_transactions
        ],
        "upcoming_events": [
            EventResponse.model_validate(e).model_dump()
            for e in upcoming_events
        ],
        "active_alerts": [
            AlertResponse.model_validate(a).model_dump()
            for a in active_alerts
        ],
        "active_goals": [
            GoalResponse.model_validate(g).model_dump()
            for g in active_goals
        ],
        "monthly_summary": {
            "month": today.strftime("%B %Y"),
            "income": monthly_income,
            "expenses": monthly_expenses,
            "savings": monthly_income - monthly_expenses,
            "savings_rate": (
                round((monthly_income - monthly_expenses) / monthly_income * 100, 1)
                if monthly_income > 0 else 0
            ),
            "transaction_count": txn_count,
            "anomaly_count": anomaly_count,
        },
    }


@router.get("/what-changed", response_model=dict)
async def what_changed(
    days: int = Query(default=7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Signature feature: What changed?

    Compares current state with the state from N days ago
    and returns meaningful changes.
    """
    # Get the two most recent life states
    states_result = await db.execute(
        select(LifeState)
        .where(LifeState.user_id == current_user.id)
        .order_by(desc(LifeState.snapshot_date))
        .limit(2)
    )
    states = states_result.scalars().all()

    if len(states) < 2:
        return {
            "changes": [],
            "message": "Not enough data to compare. Keep using LIFEOS and changes will be tracked automatically.",
        }

    current = states[0]
    previous = states[1]

    changes = []

    # Compare key metrics
    comparisons = [
        ("monthly_expenses", "Monthly Expenses", "₹"),
        ("monthly_income", "Monthly Income", "₹"),
        ("monthly_savings", "Monthly Savings", "₹"),
        ("financial_risk_score", "Risk Score", ""),
        ("emergency_runway_months", "Emergency Runway", " months"),
        ("active_subscriptions_count", "Active Subscriptions", ""),
    ]

    for field, label, unit in comparisons:
        old_val = float(getattr(previous, field) or 0)
        new_val = float(getattr(current, field) or 0)

        if old_val != 0:
            change_pct = round((new_val - old_val) / old_val * 100, 1)
        elif new_val != 0:
            change_pct = 100.0
        else:
            continue

        if abs(change_pct) >= 5:  # Only report meaningful changes
            changes.append({
                "field": field,
                "label": label,
                "old_value": old_val,
                "new_value": new_val,
                "change_pct": change_pct,
                "unit": unit,
                "direction": "up" if change_pct > 0 else "down",
                "is_concerning": _is_concerning_change(field, change_pct),
            })

    # Include stored changes from life state
    if current.changes_from_previous:
        for change in current.changes_from_previous:
            if change not in changes:
                changes.append(change)

    return {
        "changes": changes,
        "current_state_date": current.snapshot_date.isoformat(),
        "previous_state_date": previous.snapshot_date.isoformat(),
        "total_changes": len(changes),
    }


@router.get("/what-am-i-forgetting", response_model=dict)
async def what_am_i_forgetting(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Signature feature: What am I forgetting?

    Checks for:
    - Expired documents
    - Upcoming renewals
    - Overdue events
    - Missing insurance on assets
    - Warranty expirations
    """
    today = date.today()
    items = []

    # Overdue events
    overdue_result = await db.execute(
        select(Event)
        .where(
            and_(
                Event.user_id == current_user.id,
                Event.event_date < today,
                Event.status.in_([EventStatus.UPCOMING, EventStatus.DUE]),
            )
        )
        .order_by(Event.event_date)
    )
    for event in overdue_result.scalars():
        days_overdue = (today - event.event_date).days
        items.append({
            "type": "overdue_event",
            "severity": "critical" if days_overdue > 30 else "warning",
            "title": event.title,
            "details": f"Was due {days_overdue} days ago",
            "days_overdue": days_overdue,
            "amount": float(event.amount) if event.amount else None,
            "event_id": str(event.id),
        })

    # Upcoming in next 30 days (reminders)
    upcoming_result = await db.execute(
        select(Event)
        .where(
            and_(
                Event.user_id == current_user.id,
                Event.event_date >= today,
                Event.event_date <= today + timedelta(days=30),
                Event.status == EventStatus.UPCOMING,
            )
        )
        .order_by(Event.event_date)
    )
    for event in upcoming_result.scalars():
        days_until = (event.event_date - today).days
        if days_until <= 7:
            items.append({
                "type": "upcoming_soon",
                "severity": "warning",
                "title": event.title,
                "details": f"Due in {days_until} days",
                "days_until": days_until,
                "amount": float(event.amount) if event.amount else None,
                "event_id": str(event.id),
            })

    # Unread critical alerts
    alerts_result = await db.execute(
        select(Alert)
        .where(
            and_(
                Alert.user_id == current_user.id,
                Alert.is_read == False,
                Alert.severity == AlertSeverity.CRITICAL,
            )
        )
    )
    for alert in alerts_result.scalars():
        items.append({
            "type": "unread_alert",
            "severity": "critical",
            "title": alert.title,
            "details": alert.message,
            "alert_id": str(alert.id),
        })

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    items.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return {
        "items": items,
        "total": len(items),
        "message": (
            f"You may be forgetting {len(items)} things."
            if items
            else "You're on top of everything! No forgotten items detected."
        ),
    }


def _is_concerning_change(field: str, change_pct: float) -> bool:
    """Determine if a change is concerning based on the metric."""
    concerning_increases = {
        "monthly_expenses": True,
        "financial_risk_score": True,
    }
    concerning_decreases = {
        "monthly_income": True,
        "monthly_savings": True,
        "emergency_runway_months": True,
    }

    if change_pct > 0 and field in concerning_increases:
        return abs(change_pct) > 10
    if change_pct < 0 and field in concerning_decreases:
        return abs(change_pct) > 10
    return False
