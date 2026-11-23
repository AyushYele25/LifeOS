"""
LIFEOS Event Horizon Engine Tests

Validates:
1. Obligation compilation across loans, insurance, subscriptions, and assets.
2. Deterministic 30/60/90-day cash outflow requirement calculations.
3. Overdue obligation detection.
4. Recurring event completion with automatic roll-forward.
5. Events REST API lifecycle (create, list, sync, horizon, complete, delete).
"""

from datetime import date, timedelta
from decimal import Decimal
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.models import (
    User,
    Event,
    EventType,
    EventStatus,
    Loan,
    LoanType,
    LoanStatus,
    Insurance,
    InsuranceType,
    Subscription,
    SubscriptionStatus,
    SubscriptionFrequency,
    Asset,
    AssetType,
)
from app.services.event_engine import EventEngine


@pytest.mark.asyncio
async def test_sync_obligations_from_loans_and_insurance(db_session, test_user):
    """
    Verify loans and insurance policies automatically generate 90-day horizon events.
    """
    today = date.today()

    # Active Car Loan
    loan = Loan(
        user_id=test_user.id,
        name="HDFC Car Loan",
        loan_type=LoanType.VEHICLE,
        lender="HDFC Bank",
        principal_amount=Decimal("400000.00"),
        emi_amount=Decimal("8500.00"),
        emi_day=15,
        status=LoanStatus.ACTIVE,
    )
    # Active Health Insurance (Renewal in 40 days)
    insurance = Insurance(
        user_id=test_user.id,
        name="Star Health Insurance",
        insurance_type=InsuranceType.HEALTH,
        provider="Star Health",
        policy_number="STAR-992144",
        premium_amount=Decimal("15000.00"),
        expiry_date=today + timedelta(days=40),
        is_active=True,
    )
    db_session.add_all([loan, insurance])
    await db_session.commit()

    engine = EventEngine(db_session, test_user.id)
    result = await engine.sync_all_obligations(horizon_days=90)

    assert result["scanned_sources"]["loans"] == 1
    assert result["scanned_sources"]["insurance"] == 1
    assert result["new_events_created"] >= 2

    # Query created events
    events = (
        await db_session.execute(select(Event).where(Event.user_id == test_user.id))
    ).scalars().all()

    loan_events = [e for e in events if e.event_type == EventType.LOAN_EMI]
    assert len(loan_events) >= 2  # At least 2 monthly EMIs within 90 days
    assert loan_events[0].amount == Decimal("8500.00")

    ins_events = [e for e in events if e.event_type == EventType.INSURANCE_RENEWAL]
    assert len(ins_events) == 1
    assert ins_events[0].amount == Decimal("15000.00")
    assert ins_events[0].event_date == today + timedelta(days=40)


@pytest.mark.asyncio
async def test_sync_subscriptions_and_warranties(db_session, test_user):
    """
    Verify subscriptions and asset warranties generate upcoming horizon events.
    """
    today = date.today()

    sub = Subscription(
        user_id=test_user.id,
        name="Netflix 4K",
        provider="Netflix",
        amount=Decimal("649.00"),
        frequency=SubscriptionFrequency.MONTHLY,
        next_billing_date=today + timedelta(days=10),
        status=SubscriptionStatus.ACTIVE,
    )
    asset = Asset(
        user_id=test_user.id,
        name="MacBook Pro",
        asset_type=AssetType.ELECTRONICS,
        warranty_expiry=today + timedelta(days=50),
        is_active=True,
    )
    db_session.add_all([sub, asset])
    await db_session.commit()

    engine = EventEngine(db_session, test_user.id)
    result = await engine.sync_all_obligations(horizon_days=90)

    assert result["new_events_created"] >= 2

    events = (
        await db_session.execute(select(Event).where(Event.user_id == test_user.id))
    ).scalars().all()

    sub_events = [e for e in events if e.event_type == EventType.SUBSCRIPTION_PAYMENT]
    assert len(sub_events) >= 2
    assert sub_events[0].amount == Decimal("649.00")

    warranty_events = [e for e in events if e.event_type == EventType.WARRANTY_EXPIRY]
    assert len(warranty_events) == 1
    assert warranty_events[0].event_date == today + timedelta(days=50)


@pytest.mark.asyncio
async def test_overdue_event_marking(db_session, test_user):
    """
    Verify past events still in UPCOMING status are marked OVERDUE during sync.
    """
    past_date = date.today() - timedelta(days=5)
    event = Event(
        user_id=test_user.id,
        event_type=EventType.BILL_DUE,
        title="Electricity Bill Overdue",
        event_date=past_date,
        amount=Decimal("2400.00"),
        status=EventStatus.UPCOMING,
    )
    db_session.add(event)
    await db_session.commit()

    engine = EventEngine(db_session, test_user.id)
    result = await engine.sync_all_obligations(horizon_days=90)

    assert result["overdue_events_marked"] >= 1

    await db_session.refresh(event)
    assert event.status == EventStatus.OVERDUE


@pytest.mark.asyncio
async def test_deterministic_horizon_calculations(db_session, test_user):
    """
    Verify exact deterministic cash requirements across 30, 60, and 90-day intervals.
    """
    today = date.today()

    e1 = Event(
        user_id=test_user.id,
        event_type=EventType.LOAN_EMI,
        title="Loan 1",
        event_date=today + timedelta(days=10),
        amount=Decimal("10000.00"),
        status=EventStatus.UPCOMING,
    )
    e2 = Event(
        user_id=test_user.id,
        event_type=EventType.INSURANCE_RENEWAL,
        title="Insurance 1",
        event_date=today + timedelta(days=45),
        amount=Decimal("20000.00"),
        status=EventStatus.UPCOMING,
    )
    e3 = Event(
        user_id=test_user.id,
        event_type=EventType.BILL_DUE,
        title="Bill 1",
        event_date=today + timedelta(days=75),
        amount=Decimal("5000.00"),
        status=EventStatus.UPCOMING,
    )
    db_session.add_all([e1, e2, e3])
    await db_session.commit()

    engine = EventEngine(db_session, test_user.id)
    summary = await engine.get_horizon_summary(horizon_days=90)

    assert summary["totals"]["total_30d"] == Decimal("10000.00")
    assert summary["totals"]["total_60d"] == Decimal("30000.00")
    assert summary["totals"]["total_90d"] == Decimal("35000.00")

    assert summary["by_type"]["loan_emi"] == Decimal("10000.00")
    assert summary["by_type"]["insurance_renewal"] == Decimal("20000.00")
    assert summary["by_type"]["bill_due"] == Decimal("5000.00")
    assert len(summary["timeline"]) == 3


@pytest.mark.asyncio
async def test_event_completion_and_recurring_rollforward(db_session, test_user):
    """
    Verify completing a recurring EMI event rolls forward to next month's occurrence.
    """
    today = date.today()
    event = Event(
        user_id=test_user.id,
        event_type=EventType.LOAN_EMI,
        title="Car Loan EMI",
        event_date=today,
        amount=Decimal("7500.00"),
        status=EventStatus.DUE,
        is_recurring=True,
        recurrence_pattern={"frequency": "monthly", "day": today.day},
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)

    engine = EventEngine(db_session, test_user.id)
    completed_event = await engine.complete_event(event.id)

    assert completed_event.status == EventStatus.COMPLETED

    # Check that a new future event was created for next month
    events = (
        await db_session.execute(
            select(Event).where(
                Event.user_id == test_user.id, Event.status == EventStatus.UPCOMING
            )
        )
    ).scalars().all()

    assert len(events) == 1
    next_event = events[0]
    assert next_event.amount == Decimal("7500.00")
    assert next_event.event_date > today


@pytest.mark.asyncio
async def test_events_api_workflow(client: AsyncClient, test_user: User, auth_headers: dict):
    """
    Tests REST API lifecycle:
    1. POST /api/events (create custom event)
    2. GET /api/events (list upcoming)
    3. GET /api/events/horizon (get horizon summary)
    4. PATCH /api/events/{id}/complete (mark complete)
    5. DELETE /api/events/{id} (delete event)
    """
    target_d = str(date.today() + timedelta(days=20))

    # 1. Create custom event
    create_resp = await client.post(
        "/api/events",
        headers=auth_headers,
        json={
            "event_type": "custom",
            "title": "Annual Property Tax Payment",
            "description": "BBMP Property Tax Koramangala",
            "event_date": target_d,
            "amount": 18500.0,
            "is_recurring": False,
        },
    )
    assert create_resp.status_code == 201
    event_data = create_resp.json()
    event_id = event_data["id"]
    assert event_data["title"] == "Annual Property Tax Payment"

    # 2. List events
    list_resp = await client.get("/api/events?days_ahead=30", headers=auth_headers)
    assert list_resp.status_code == 200
    events_list = list_resp.json()
    assert len(events_list) >= 1
    assert any(e["id"] == event_id for e in events_list)

    # 3. Get horizon summary
    horiz_resp = await client.get("/api/events/horizon?horizon_days=90", headers=auth_headers)
    assert horiz_resp.status_code == 200
    horiz_data = horiz_resp.json()
    assert float(horiz_data["totals"]["total_30d"]) >= 18500.0

    # 4. Complete event
    comp_resp = await client.patch(f"/api/events/{event_id}/complete", headers=auth_headers)
    assert comp_resp.status_code == 200
    assert comp_resp.json()["status"] == "completed"

    # 5. Delete event
    del_resp = await client.delete(f"/api/events/{event_id}", headers=auth_headers)
    assert del_resp.status_code == 204
