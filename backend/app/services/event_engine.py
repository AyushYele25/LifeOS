"""
LIFEOS Event Horizon Engine

Compiles and manages the forward-looking obligation timeline:
- Confirmed loan EMIs
- Insurance premium renewals
- Active recurring subscriptions
- Warranty expirations & vehicle service schedules
- Deterministic 30/60/90-day cash outflow aggregations
- Automated roll-forward for recurring obligations
"""

from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import uuid

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Event,
    EventType,
    EventStatus,
    Loan,
    LoanStatus,
    Insurance,
    Subscription,
    SubscriptionStatus,
    SubscriptionFrequency,
    Asset,
    AuditLog,
)


class EventEngine:
    """
    Stateful engine that maintains and projects future household obligations.
    """

    def __init__(self, db: AsyncSession, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id

    async def sync_all_obligations(self, horizon_days: int = 90) -> Dict[str, Any]:
        """
        Reconciles all active loans, insurance policies, subscriptions, and assets,
        ensuring every contractually confirmed upcoming obligation has a corresponding Event.
        Also marks any past upcoming events as OVERDUE.
        """
        today = date.today()
        horizon_end = today + timedelta(days=horizon_days)

        scanned_counts = {
            "loans": 0,
            "insurance": 0,
            "subscriptions": 0,
            "assets": 0,
        }
        new_events: List[Event] = []

        # 1. Fetch existing future events to avoid duplicates
        existing_res = await self.db.execute(
            select(Event).where(
                and_(
                    Event.user_id == self.user_id,
                    Event.event_date >= today - timedelta(days=365),
                )
            )
        )
        existing_events = existing_res.scalars().all()
        existing_keys = {
            (e.source_type, str(e.source_id) if e.source_id else "", e.event_date)
            for e in existing_events
        }

        # 2. Sync Active Loans
        loan_res = await self.db.execute(
            select(Loan).where(
                and_(Loan.user_id == self.user_id, Loan.status == LoanStatus.ACTIVE)
            )
        )
        loans = loan_res.scalars().all()
        scanned_counts["loans"] = len(loans)

        for loan in loans:
            if not loan.emi_amount:
                continue
            day = loan.emi_day or (loan.start_date.day if loan.start_date else 10)
            day = min(day, 28)

            # Generate monthly EMIs up to horizon_end
            curr = today
            while curr <= horizon_end:
                # Target due date in curr's month
                target_date = date(curr.year, curr.month, day)
                if target_date < today:
                    # Move to next month
                    curr = (curr.replace(day=1) + timedelta(days=32)).replace(day=1)
                    continue

                if target_date <= horizon_end:
                    key = ("loan", str(loan.id), target_date)
                    if key not in existing_keys:
                        event = Event(
                            user_id=self.user_id,
                            event_type=EventType.LOAN_EMI,
                            title=f"Loan EMI: {loan.name}",
                            description=f"Monthly EMI payment for {loan.name}",
                            event_date=target_date,
                            amount=loan.emi_amount,
                            status=EventStatus.UPCOMING,
                            confidence=1.0,
                            source_type="loan",
                            source_id=loan.id,
                            is_recurring=True,
                            recurrence_pattern={"frequency": "monthly", "day": day},
                        )
                        self.db.add(event)
                        new_events.append(event)
                        existing_keys.add(key)

                # Advance one month
                curr = (curr.replace(day=1) + timedelta(days=32)).replace(day=1)

        # 3. Sync Insurance Policies
        ins_res = await self.db.execute(
            select(Insurance).where(
                and_(Insurance.user_id == self.user_id, Insurance.is_active == True)
            )
        )
        policies = ins_res.scalars().all()
        scanned_counts["insurance"] = len(policies)

        for pol in policies:
            renewal_d = pol.next_premium_date or pol.expiry_date
            if renewal_d and today <= renewal_d <= horizon_end:
                key = ("insurance", str(pol.id), renewal_d)
                if key not in existing_keys:
                    event = Event(
                        user_id=self.user_id,
                        event_type=EventType.INSURANCE_RENEWAL,
                        title=f"Insurance Renewal: {pol.provider or pol.name}",
                        description=f"Premium renewal for policy {pol.policy_number or pol.name}",
                        event_date=renewal_d,
                        amount=pol.premium_amount,
                        status=EventStatus.UPCOMING,
                        confidence=1.0,
                        source_type="insurance",
                        source_id=pol.id,
                        is_recurring=True,
                        recurrence_pattern={"frequency": "yearly"},
                    )
                    self.db.add(event)
                    new_events.append(event)
                    existing_keys.add(key)

        # 4. Sync Active Subscriptions
        sub_res = await self.db.execute(
            select(Subscription).where(
                and_(
                    Subscription.user_id == self.user_id,
                    Subscription.status == SubscriptionStatus.ACTIVE,
                )
            )
        )
        subs = sub_res.scalars().all()
        scanned_counts["subscriptions"] = len(subs)

        for sub in subs:
            if not sub.amount:
                continue
            next_bill = sub.next_billing_date or today
            curr = next_bill if next_bill >= today else today

            interval_days = 30
            if sub.frequency == SubscriptionFrequency.WEEKLY:
                interval_days = 7
            elif sub.frequency == SubscriptionFrequency.QUARTERLY:
                interval_days = 90
            elif sub.frequency == SubscriptionFrequency.YEARLY:
                interval_days = 365

            while curr <= horizon_end:
                key = ("subscription", str(sub.id), curr)
                if key not in existing_keys:
                    event = Event(
                        user_id=self.user_id,
                        event_type=EventType.SUBSCRIPTION_PAYMENT,
                        title=f"Subscription: {sub.name}",
                        description=f"Recurring {sub.frequency.value} payment to {sub.provider or sub.name}",
                        event_date=curr,
                        amount=sub.amount,
                        status=EventStatus.UPCOMING,
                        confidence=0.95,
                        source_type="subscription",
                        source_id=sub.id,
                        is_recurring=True,
                        recurrence_pattern={"frequency": sub.frequency.value},
                    )
                    self.db.add(event)
                    new_events.append(event)
                    existing_keys.add(key)
                curr += timedelta(days=interval_days)

        # 5. Sync Assets (Warranties & Scheduled Services)
        asset_res = await self.db.execute(
            select(Asset).where(
                and_(Asset.user_id == self.user_id, Asset.is_active == True)
            )
        )
        assets = asset_res.scalars().all()
        scanned_counts["assets"] = len(assets)

        for ast in assets:
            # Warranty expiry
            if ast.warranty_expiry and today <= ast.warranty_expiry <= horizon_end:
                key = ("asset", str(ast.id), ast.warranty_expiry)
                if key not in existing_keys:
                    event = Event(
                        user_id=self.user_id,
                        event_type=EventType.WARRANTY_EXPIRY,
                        title=f"Warranty Expiry: {ast.name}",
                        description=f"Manufacturer warranty expires for {ast.name}",
                        event_date=ast.warranty_expiry,
                        amount=Decimal("0.00"),
                        status=EventStatus.UPCOMING,
                        confidence=1.0,
                        source_type="asset",
                        source_id=ast.id,
                    )
                    self.db.add(event)
                    new_events.append(event)
                    existing_keys.add(key)

            # Next service date
            if ast.next_service_date and today <= ast.next_service_date <= horizon_end:
                key = ("service", str(ast.id), ast.next_service_date)
                if key not in existing_keys:
                    event = Event(
                        user_id=self.user_id,
                        event_type=EventType.VEHICLE_SERVICE,
                        title=f"Service Due: {ast.name}",
                        description=f"Scheduled maintenance due for {ast.name}",
                        event_date=ast.next_service_date,
                        amount=Decimal("3500.00"),
                        status=EventStatus.UPCOMING,
                        confidence=0.85,
                        source_type="asset",
                        source_id=ast.id,
                    )
                    self.db.add(event)
                    new_events.append(event)
                    existing_keys.add(key)

        # 6. Mark Overdue Events
        overdue_res = await self.db.execute(
            select(Event).where(
                and_(
                    Event.user_id == self.user_id,
                    Event.event_date < today,
                    Event.status == EventStatus.UPCOMING,
                )
            )
        )
        overdue_events = overdue_res.scalars().all()
        for oe in overdue_events:
            oe.status = EventStatus.OVERDUE

        # Audit log
        audit = AuditLog(
            user_id=self.user_id,
            action="event_horizon_sync",
            entity_type="event",
            details={
                "scanned_sources": scanned_counts,
                "new_events_created": len(new_events),
                "overdue_marked": len(overdue_events),
            },
        )
        self.db.add(audit)
        await self.db.commit()

        for ne in new_events:
            await self.db.refresh(ne)

        return {
            "scanned_sources": scanned_counts,
            "new_events_created": len(new_events),
            "overdue_events_marked": len(overdue_events),
            "events": new_events,
        }

    async def get_horizon_summary(
        self, as_of: Optional[date] = None, horizon_days: int = 90
    ) -> Dict[str, Any]:
        """
        Calculates exact deterministic 30/60/90-day cash outflow requirements,
        grouped by type, month, and chronological timeline.
        """
        ref_date = as_of or date.today()
        horizon_end = ref_date + timedelta(days=horizon_days)

        query = (
            select(Event)
            .where(
                and_(
                    Event.user_id == self.user_id,
                    Event.event_date >= ref_date,
                    Event.event_date <= horizon_end,
                    Event.status.in_([EventStatus.UPCOMING, EventStatus.DUE, EventStatus.OVERDUE]),
                )
            )
            .order_by(Event.event_date.asc())
        )
        res = await self.db.execute(query)
        events = res.scalars().all()

        total_30d = Decimal("0.00")
        total_60d = Decimal("0.00")
        total_90d = Decimal("0.00")

        by_type: Dict[str, Decimal] = {}
        by_month: Dict[str, List[Event]] = {}

        upcoming_count = 0
        overdue_count = 0

        for e in events:
            amt = e.amount or Decimal("0.00")
            delta_days = (e.event_date - ref_date).days
            e.days_until = delta_days

            if e.status == EventStatus.OVERDUE:
                overdue_count += 1
            else:
                upcoming_count += 1

            if delta_days <= 30:
                total_30d += amt
            if delta_days <= 60:
                total_60d += amt
            if delta_days <= 90:
                total_90d += amt

            # By type breakdown
            t_key = e.event_type.value
            by_type[t_key] = by_type.get(t_key, Decimal("0.00")) + amt

            # By month breakdown
            m_key = e.event_date.strftime("%B %Y")
            if m_key not in by_month:
                by_month[m_key] = []
            by_month[m_key].append(e)

        return {
            "as_of_date": ref_date,
            "totals": {
                "total_30d": total_30d,
                "total_60d": total_60d,
                "total_90d": total_90d,
            },
            "by_type": by_type,
            "by_month": by_month,
            "upcoming_events_count": upcoming_count,
            "overdue_events_count": overdue_count,
            "timeline": events,
        }

    async def complete_event(self, event_id: uuid.UUID) -> Optional[Event]:
        """
        Marks an event as COMPLETED.
        If the event is marked is_recurring, automatically generates the next period's Event.
        """
        res = await self.db.execute(
            select(Event).where(
                and_(Event.id == event_id, Event.user_id == self.user_id)
            )
        )
        event = res.scalar_one_or_none()
        if not event:
            return None

        event.status = EventStatus.COMPLETED

        # Roll forward if recurring
        if event.is_recurring and event.recurrence_pattern:
            freq = event.recurrence_pattern.get("frequency", "monthly")
            next_date = event.event_date

            if freq == "weekly":
                next_date += timedelta(days=7)
            elif freq == "quarterly":
                next_date += timedelta(days=90)
            elif freq == "yearly":
                next_date = next_date.replace(year=next_date.year + 1)
            else:  # monthly
                day = event.recurrence_pattern.get("day", event.event_date.day)
                m = event.event_date.month + 1
                y = event.event_date.year + (1 if m > 12 else 0)
                m = 1 if m > 12 else m
                next_date = date(y, m, min(day, 28))

            next_event = Event(
                user_id=self.user_id,
                event_type=event.event_type,
                title=event.title,
                description=event.description,
                event_date=next_date,
                amount=event.amount,
                status=EventStatus.UPCOMING,
                confidence=event.confidence,
                source_type=event.source_type,
                source_id=event.source_id,
                is_recurring=True,
                recurrence_pattern=event.recurrence_pattern,
            )
            self.db.add(next_event)

        audit = AuditLog(
            user_id=self.user_id,
            action="event_completed",
            entity_type="event",
            entity_id=str(event.id),
            details={"is_recurring": event.is_recurring},
        )
        self.db.add(audit)
        await self.db.commit()
        await self.db.refresh(event)
        return event
