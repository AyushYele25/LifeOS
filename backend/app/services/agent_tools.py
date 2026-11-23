"""
LIFEOS Agentic AI — Typed Tool Definitions

13 deterministic, read-only tools that give the agentic layer access to every
LIFEOS intelligence engine.  Each tool declares:
  • name          – Unique identifier used in tool-call routing.
  • description   – Human-readable docstring the planner sees when choosing tools.
  • parameters    – JSON-Schema dict describing required/optional inputs.
  • execute(db, user_id, params) → dict  – Async function that runs the tool.

Safety: Every tool is READ-ONLY.  No writes, no deletes, no mutations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.models import (
    Account,
    AccountType,
    Transaction,
    TransactionType,
    Category,
    Loan,
    LoanStatus,
    Insurance,
    Subscription,
    SubscriptionStatus,
    Event,
    EventStatus,
    Alert,
    RecurringPattern,
    RiskLevel,
)
from app.services.life_state_engine import LifeStateEngine
from app.services.event_engine import EventEngine
from app.services.forecast_engine import ForecastEngine
from app.services.risk_engine import RiskEngine
from app.services.scenario_simulator import ScenarioSimulator
from app.services.anomaly_detector import AnomalyDetector
from app.services.recurring_detector import RecurringDetector
from app.services.rag_service import RAGService, rag_service


# ═══════════════════════════════════════════════════════
# Tool Descriptor
# ═══════════════════════════════════════════════════════

@dataclass
class ToolDescriptor:
    """Schema-carrying definition of a single agent tool."""
    name: str
    description: str
    parameters: Dict[str, Any]          # JSON-Schema subset
    execute: Callable[..., Coroutine]    # async (db, user_id, params) → dict
    category: str = "general"


# ═══════════════════════════════════════════════════════
# Helper — safe decimal serialisation
# ═══════════════════════════════════════════════════════

def _dec(v) -> float:
    """Convert Decimal / None → float for JSON-safe responses."""
    if v is None:
        return 0.0
    return float(v)


def _serialize_date(v) -> Optional[str]:
    """Convert date / datetime → ISO string or None."""
    if v is None:
        return None
    return str(v)


# ═══════════════════════════════════════════════════════
# Tool 1: get_life_state
# ═══════════════════════════════════════════════════════

async def _exec_get_life_state(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the user's current deterministic financial snapshot."""
    engine = LifeStateEngine(db, user_id)
    state = await engine.get_current_state()
    return {
        "total_balance": _dec(state.total_balance),
        "monthly_income": _dec(state.monthly_income),
        "monthly_expenses": _dec(state.monthly_expenses),
        "monthly_savings": _dec(state.monthly_savings),
        "savings_rate": round(state.savings_rate or 0, 2),
        "emergency_runway_months": round(state.emergency_runway_months or 0, 1),
        "upcoming_obligations_30d": _dec(state.upcoming_obligations_30d),
        "upcoming_obligations_60d": _dec(state.upcoming_obligations_60d),
        "upcoming_obligations_90d": _dec(state.upcoming_obligations_90d),
        "active_loans_count": state.active_loans_count or 0,
        "total_emi_burden": _dec(state.total_emi_burden),
        "active_subscriptions_count": state.active_subscriptions_count or 0,
        "total_subscription_burden": _dec(state.total_subscription_burden),
        "financial_risk_score": round(state.financial_risk_score or 0, 1),
        "risk_level": (state.risk_level.value if state.risk_level else "low"),
        "expense_by_category": state.expense_by_category or {},
        "income_by_source": state.income_by_source or {},
    }


TOOL_GET_LIFE_STATE = ToolDescriptor(
    name="get_life_state",
    description=(
        "Returns the user's complete deterministic financial snapshot: "
        "liquid balance, monthly income & expenses, savings rate, emergency "
        "runway months, upcoming obligations (30/60/90 days), active loans & "
        "subscriptions, risk score, and expense/income breakdowns."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_exec_get_life_state,
    category="state",
)


# ═══════════════════════════════════════════════════════
# Tool 2: get_transactions
# ═══════════════════════════════════════════════════════

async def _exec_get_transactions(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Query user transactions with optional filters."""
    query = (
        select(Transaction)
        .options(selectinload(Transaction.category))
        .where(Transaction.user_id == user_id)
    )

    if params.get("start_date"):
        query = query.where(Transaction.date >= date.fromisoformat(params["start_date"]))
    if params.get("end_date"):
        query = query.where(Transaction.date <= date.fromisoformat(params["end_date"]))
    if params.get("transaction_type"):
        query = query.where(Transaction.transaction_type == params["transaction_type"])
    if params.get("merchant"):
        query = query.where(Transaction.merchant.ilike(f"%{params['merchant']}%"))
    if params.get("min_amount"):
        query = query.where(Transaction.amount >= Decimal(str(params["min_amount"])))
    if params.get("max_amount"):
        query = query.where(Transaction.amount <= Decimal(str(params["max_amount"])))

    limit = min(int(params.get("limit", 20)), 50)
    query = query.order_by(desc(Transaction.date)).limit(limit)

    result = await db.execute(query)
    txns = result.scalars().all()

    return {
        "count": len(txns),
        "transactions": [
            {
                "id": str(t.id),
                "date": str(t.date),
                "description": t.description,
                "amount": _dec(t.amount),
                "type": t.transaction_type.value,
                "merchant": t.merchant,
                "category": t.category.name if t.category else None,
                "is_recurring": t.is_recurring,
                "is_anomaly": t.is_anomaly,
            }
            for t in txns
        ],
    }


TOOL_GET_TRANSACTIONS = ToolDescriptor(
    name="get_transactions",
    description=(
        "Query the user's transaction history with optional filters: "
        "start_date, end_date, transaction_type (credit/debit), merchant, "
        "min_amount, max_amount, limit (max 50). Returns recent transactions "
        "with date, description, amount, category, and flags."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "transaction_type": {"type": "string", "enum": ["credit", "debit"]},
            "merchant": {"type": "string", "description": "Merchant name substring"},
            "min_amount": {"type": "number"},
            "max_amount": {"type": "number"},
            "limit": {"type": "integer", "description": "Max results (default 20, max 50)"},
        },
        "required": [],
    },
    execute=_exec_get_transactions,
    category="transactions",
)


# ═══════════════════════════════════════════════════════
# Tool 3: get_spending_by_category
# ═══════════════════════════════════════════════════════

async def _exec_get_spending_by_category(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate spending grouped by category over a date range."""
    days = int(params.get("days", 30))
    cutoff = date.today() - timedelta(days=days)

    query = (
        select(
            Category.name,
            func.sum(Transaction.amount).label("total"),
            func.count(Transaction.id).label("count"),
        )
        .join(Category, Transaction.category_id == Category.id)
        .where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.DEBIT,
                Transaction.date >= cutoff,
            )
        )
        .group_by(Category.name)
        .order_by(desc("total"))
    )
    result = await db.execute(query)
    rows = result.all()

    breakdown = [
        {"category": r[0], "total_spent": _dec(r[1]), "transaction_count": r[2]}
        for r in rows
    ]
    grand_total = sum(item["total_spent"] for item in breakdown)

    return {
        "period_days": days,
        "total_spent": round(grand_total, 2),
        "categories": breakdown,
    }


TOOL_GET_SPENDING_BY_CATEGORY = ToolDescriptor(
    name="get_spending_by_category",
    description=(
        "Aggregates user spending grouped by transaction category over a "
        "configurable number of days (default 30). Returns each category's "
        "total spend and transaction count, sorted highest-first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Look-back window in days (default 30)",
            },
        },
        "required": [],
    },
    execute=_exec_get_spending_by_category,
    category="transactions",
)


# ═══════════════════════════════════════════════════════
# Tool 4: get_recurring_payments
# ═══════════════════════════════════════════════════════

async def _exec_get_recurring_payments(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return detected recurring payment patterns."""
    query = (
        select(RecurringPattern)
        .where(RecurringPattern.user_id == user_id)
        .order_by(desc(RecurringPattern.confidence))
    )
    result = await db.execute(query)
    patterns = result.scalars().all()

    return {
        "count": len(patterns),
        "patterns": [
            {
                "id": str(p.id),
                "merchant": p.merchant,
                "category": p.category,
                "average_amount": _dec(p.average_amount),
                "frequency": p.frequency.value if p.frequency else None,
                "next_expected_date": _serialize_date(p.next_expected_date),
                "confidence": round(p.confidence, 2),
                "transaction_count": p.transaction_count,
                "is_confirmed": p.is_confirmed,
            }
            for p in patterns
        ],
    }


TOOL_GET_RECURRING_PAYMENTS = ToolDescriptor(
    name="get_recurring_payments",
    description=(
        "Returns all detected recurring payment patterns — subscriptions, EMIs, "
        "utility bills — with merchant name, average amount, frequency, next "
        "expected date, and confidence score."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_exec_get_recurring_payments,
    category="recurring",
)


# ═══════════════════════════════════════════════════════
# Tool 5: get_upcoming_events
# ═══════════════════════════════════════════════════════

async def _exec_get_upcoming_events(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return upcoming obligation events within horizon."""
    horizon_days = int(params.get("horizon_days", 90))
    today = date.today()
    cutoff = today + timedelta(days=horizon_days)

    query = (
        select(Event)
        .where(
            and_(
                Event.user_id == user_id,
                Event.event_date >= today,
                Event.event_date <= cutoff,
                Event.status.in_([EventStatus.UPCOMING, EventStatus.DUE]),
            )
        )
        .order_by(Event.event_date.asc())
    )
    result = await db.execute(query)
    events = result.scalars().all()

    total_amount = sum(_dec(e.amount) for e in events if e.amount)

    return {
        "horizon_days": horizon_days,
        "event_count": len(events),
        "total_amount": round(total_amount, 2),
        "events": [
            {
                "id": str(e.id),
                "type": e.event_type.value,
                "title": e.title,
                "date": str(e.event_date),
                "amount": _dec(e.amount),
                "status": e.status.value,
                "days_until": (e.event_date - today).days,
                "source_type": e.source_type,
            }
            for e in events
        ],
    }


TOOL_GET_UPCOMING_EVENTS = ToolDescriptor(
    name="get_upcoming_events",
    description=(
        "Returns upcoming financial obligations within a configurable horizon "
        "(default 90 days): loan EMIs, insurance renewals, subscription billings, "
        "warranty expirations, and bill due dates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "horizon_days": {
                "type": "integer",
                "description": "Number of days ahead to look (default 90, max 365)",
            },
        },
        "required": [],
    },
    execute=_exec_get_upcoming_events,
    category="events",
)


# ═══════════════════════════════════════════════════════
# Tool 6: get_alerts
# ═══════════════════════════════════════════════════════

async def _exec_get_alerts(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return active (un-dismissed) alerts and anomalies."""
    query = (
        select(Alert)
        .where(
            and_(
                Alert.user_id == user_id,
                Alert.is_dismissed == False,
            )
        )
        .order_by(desc(Alert.created_at))
        .limit(int(params.get("limit", 20)))
    )
    result = await db.execute(query)
    alerts = result.scalars().all()

    return {
        "count": len(alerts),
        "alerts": [
            {
                "id": str(a.id),
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message,
                "is_read": a.is_read,
                "created_at": _serialize_date(a.created_at),
            }
            for a in alerts
        ],
    }


TOOL_GET_ALERTS = ToolDescriptor(
    name="get_alerts",
    description=(
        "Returns the user's active (un-dismissed) alerts and anomaly "
        "notifications, including duplicates, spend spikes, salary drops, "
        "and subscription price creep. Each alert has type, severity, title, "
        "and evidence message."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max alerts (default 20)"},
        },
        "required": [],
    },
    execute=_exec_get_alerts,
    category="alerts",
)


# ═══════════════════════════════════════════════════════
# Tool 7: get_risk_assessment
# ═══════════════════════════════════════════════════════

async def _exec_get_risk_assessment(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the 5-pillar financial risk assessment with stress tests."""
    engine = RiskEngine(db, user_id)
    assessment = await engine.get_assessment()

    # assessment is a dict from RiskEngine.get_assessment()
    return {
        "composite_risk_score": assessment.get("composite_risk_score", 0),
        "risk_level": assessment.get("risk_level", "low"),
        "risk_grade": assessment.get("risk_grade", "N/A"),
        "dti_ratio": assessment.get("dti_ratio", 0),
        "savings_rate": assessment.get("savings_rate", 0),
        "emergency_runway_months": assessment.get("emergency_runway_months", 0),
        "vulnerabilities": assessment.get("vulnerabilities", []),
        "recommendations": assessment.get("recommendations", []),
        "breakdown": assessment.get("breakdown", []),
    }


TOOL_GET_RISK_ASSESSMENT = ToolDescriptor(
    name="get_risk_assessment",
    description=(
        "Runs the comprehensive 5-pillar financial risk assessment: Liquidity, "
        "Debt-to-Income, Cashflow, Obligations, and Concentration. Returns a "
        "composite risk score (0-100), risk grade (A-F), vulnerabilities, "
        "actionable recommendations, and stress test results."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_exec_get_risk_assessment,
    category="risk",
)


# ═══════════════════════════════════════════════════════
# Tool 8: forecast_expenses
# ═══════════════════════════════════════════════════════

async def _exec_forecast_expenses(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Produce 30/60/90-day expense forecasts with confidence intervals."""
    engine = ForecastEngine(db, user_id)
    forecast = await engine.generate_forecast()

    def _period(p: dict) -> dict:
        return {
            "days": p.get("days", 0),
            "total_floor": _dec(p.get("total_floor")),
            "p10_lower": _dec(p.get("p10_lower")),
            "p50_expected": _dec(p.get("p50_expected")),
            "p90_upper": _dec(p.get("p90_upper")),
            "top_categories": [
                {
                    "category": c.get("category", ""),
                    "p50_expected": _dec(c.get("p50_expected")),
                }
                for c in (p.get("categories") or [])[:5]
            ],
        }

    return {
        "horizon_30d": _period(forecast.get("horizon_30d", {})),
        "horizon_60d": _period(forecast.get("horizon_60d", {})),
        "horizon_90d": _period(forecast.get("horizon_90d", {})),
        "methodology": forecast.get("methodology", ""),
        "active_seasonal_factors": forecast.get("active_seasonal_factors", []),
    }


TOOL_FORECAST_EXPENSES = ToolDescriptor(
    name="forecast_expenses",
    description=(
        "Projects the user's future expenses over 30, 60, and 90-day horizons "
        "using weighted moving averages, contractual floors, and Indian seasonal "
        "adjustments. Returns P10/P50/P90 confidence intervals per period and "
        "top-5 category breakdowns."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_exec_forecast_expenses,
    category="forecast",
)


# ═══════════════════════════════════════════════════════
# Tool 9: simulate_purchase
# ═══════════════════════════════════════════════════════

async def _exec_simulate_purchase(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a 'Can I afford X?' deterministic purchase simulation."""
    item = params.get("item", "Item")
    amount = float(params.get("amount", 0))
    emi_months = int(params.get("emi_months", 12))

    engine = ScenarioSimulator(db, user_id)
    sim = await engine.run_simulation(
        simulation_type="purchase",
        title=f"Can I afford {item}?",
        params={"item": item, "amount": amount, "emi_months": emi_months},
        persist=False,  # Agent queries are read-only; don't persist
    )

    # Simulation returns a Simulation ORM object — extract the scenarios
    scenarios_raw = sim.scenarios
    if isinstance(scenarios_raw, str):
        import json
        scenarios_raw = json.loads(scenarios_raw)

    scenarios = []
    for s in (scenarios_raw or []):
        scenarios.append({
            "name": s.get("name", ""),
            "description": s.get("description", ""),
            "savings_after": _dec(s.get("savings_after")),
            "monthly_cashflow_impact": _dec(s.get("monthly_cashflow_impact")),
            "risk_level_after": s.get("risk_level_after", ""),
            "emergency_runway_months": round(float(s.get("emergency_runway_months", 0)), 1),
        })

    return {
        "item": item,
        "amount": amount,
        "scenarios": scenarios,
        "recommendation": sim.recommendation or "",
    }


TOOL_SIMULATE_PURCHASE = ToolDescriptor(
    name="simulate_purchase",
    description=(
        "Simulates 'Can I afford X?' — compares paying upfront (cash), "
        "financing via EMI installments, and delaying to save up. Returns "
        "the impact on liquid balance, emergency runway, monthly cashflow, "
        "and risk level for each scenario, with a deterministic recommendation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "item": {"type": "string", "description": "Name of the item to purchase"},
            "amount": {"type": "number", "description": "Purchase price in INR"},
            "emi_months": {
                "type": "integer",
                "description": "EMI tenure in months (default 12)",
            },
        },
        "required": ["item", "amount"],
    },
    execute=_exec_simulate_purchase,
    category="simulator",
)


# ═══════════════════════════════════════════════════════
# Tool 10: search_documents
# ═══════════════════════════════════════════════════════

async def _exec_search_documents(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Semantic search across user's uploaded documents."""
    query_text = params.get("query", "")
    top_k = int(params.get("top_k", 5))

    try:
        service = RAGService(db, user_id)
        from app.schemas.schemas import RAGQueryRequest
        req = RAGQueryRequest(query=query_text, top_k=top_k)
        result = await service.query(req)
        return {
            "query": result.query,
            "answer": result.answer,
            "citations": [
                {
                    "filename": c.filename,
                    "document_type": c.document_type,
                    "page_number": c.page_number,
                    "text_snippet": c.text_snippet[:200],
                    "relevance_score": round(c.relevance_score, 3),
                }
                for c in result.citations
            ],
            "has_conflicts": result.has_conflicts,
            "conflict_details": result.conflict_details,
        }
    except Exception:
        return {
            "query": query_text,
            "answer": "No documents indexed or search unavailable.",
            "citations": [],
            "has_conflicts": False,
            "conflict_details": None,
        }


TOOL_SEARCH_DOCUMENTS = ToolDescriptor(
    name="search_documents",
    description=(
        "Performs semantic search over the user's uploaded documents (insurance "
        "policies, loan agreements, warranties, bills, salary slips). Returns "
        "a synthesised answer with source citations including filenames, page "
        "numbers, and text snippets."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The semantic question to search for",
            },
            "top_k": {
                "type": "integer",
                "description": "Max chunks to retrieve (default 5)",
            },
        },
        "required": ["query"],
    },
    execute=_exec_search_documents,
    category="documents",
)


# ═══════════════════════════════════════════════════════
# Tool 11: get_loans
# ═══════════════════════════════════════════════════════

async def _exec_get_loans(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return user's active loans and EMI details."""
    status_filter = params.get("status", "active")
    query = select(Loan).where(Loan.user_id == user_id)
    if status_filter == "active":
        query = query.where(Loan.status == LoanStatus.ACTIVE)
    query = query.order_by(desc(Loan.created_at))
    result = await db.execute(query)
    loans = result.scalars().all()

    total_outstanding = sum(_dec(l.outstanding_amount) for l in loans)
    total_emi = sum(_dec(l.emi_amount) for l in loans)

    return {
        "count": len(loans),
        "total_outstanding": round(total_outstanding, 2),
        "total_monthly_emi": round(total_emi, 2),
        "loans": [
            {
                "id": str(l.id),
                "name": l.name,
                "type": l.loan_type.value,
                "lender": l.lender,
                "principal": _dec(l.principal_amount),
                "outstanding": _dec(l.outstanding_amount),
                "interest_rate": l.interest_rate,
                "emi_amount": _dec(l.emi_amount),
                "emi_day": l.emi_day,
                "emis_paid": l.emis_paid,
                "total_emis": l.total_emis,
                "status": l.status.value,
                "start_date": _serialize_date(l.start_date),
                "end_date": _serialize_date(l.end_date),
            }
            for l in loans
        ],
    }


TOOL_GET_LOANS = ToolDescriptor(
    name="get_loans",
    description=(
        "Returns the user's loans with full details: principal, outstanding "
        "balance, interest rate, EMI amount, EMIs paid/remaining, lender, "
        "start and end dates. Defaults to active loans only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "closed", "all"],
                "description": "Loan status filter (default active)",
            },
        },
        "required": [],
    },
    execute=_exec_get_loans,
    category="loans",
)


# ═══════════════════════════════════════════════════════
# Tool 12: get_insurance
# ═══════════════════════════════════════════════════════

async def _exec_get_insurance(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return user's insurance policies."""
    query = (
        select(Insurance)
        .where(and_(Insurance.user_id == user_id, Insurance.is_active == True))
        .order_by(Insurance.expiry_date.asc())
    )
    result = await db.execute(query)
    policies = result.scalars().all()

    today = date.today()
    return {
        "count": len(policies),
        "policies": [
            {
                "id": str(p.id),
                "name": p.name,
                "type": p.insurance_type.value,
                "provider": p.provider,
                "policy_number": p.policy_number,
                "premium_amount": _dec(p.premium_amount),
                "premium_frequency": p.premium_frequency.value if p.premium_frequency else None,
                "sum_insured": _dec(p.sum_insured),
                "expiry_date": _serialize_date(p.expiry_date),
                "next_premium_date": _serialize_date(p.next_premium_date),
                "days_until_expiry": (
                    (p.expiry_date - today).days if p.expiry_date else None
                ),
            }
            for p in policies
        ],
    }


TOOL_GET_INSURANCE = ToolDescriptor(
    name="get_insurance",
    description=(
        "Returns the user's active insurance policies: type, provider, "
        "premium amount and frequency, sum insured, expiry date, next "
        "premium due date, and days until expiry."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_exec_get_insurance,
    category="insurance",
)


# ═══════════════════════════════════════════════════════
# Tool 13: get_subscriptions
# ═══════════════════════════════════════════════════════

async def _exec_get_subscriptions(
    db: AsyncSession, user_id: uuid.UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return user's active subscriptions."""
    query = (
        select(Subscription)
        .where(
            and_(
                Subscription.user_id == user_id,
                Subscription.status == SubscriptionStatus.ACTIVE,
            )
        )
        .order_by(Subscription.next_billing_date.asc())
    )
    result = await db.execute(query)
    subs = result.scalars().all()

    total_monthly = Decimal("0")
    for s in subs:
        amt = s.amount or Decimal("0")
        freq = s.frequency
        if freq == "yearly":
            total_monthly += amt / 12
        elif freq == "half_yearly":
            total_monthly += amt / 6
        elif freq == "quarterly":
            total_monthly += amt / 3
        elif freq == "weekly":
            total_monthly += amt * Decimal("4.33")
        elif freq == "daily":
            total_monthly += amt * Decimal("30")
        else:
            total_monthly += amt

    return {
        "count": len(subs),
        "total_monthly_cost": round(_dec(total_monthly), 2),
        "subscriptions": [
            {
                "id": str(s.id),
                "name": s.name,
                "provider": s.provider,
                "amount": _dec(s.amount),
                "frequency": s.frequency.value if s.frequency else None,
                "category": s.category,
                "next_billing_date": _serialize_date(s.next_billing_date),
                "is_auto_detected": s.is_auto_detected,
            }
            for s in subs
        ],
    }


TOOL_GET_SUBSCRIPTIONS = ToolDescriptor(
    name="get_subscriptions",
    description=(
        "Returns the user's active subscriptions (Netflix, Spotify, Gym, etc.) "
        "with amount, billing frequency, next billing date, and normalised "
        "total monthly cost across all subscriptions."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_exec_get_subscriptions,
    category="subscriptions",
)


# ═══════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, ToolDescriptor] = {
    t.name: t
    for t in [
        TOOL_GET_LIFE_STATE,
        TOOL_GET_TRANSACTIONS,
        TOOL_GET_SPENDING_BY_CATEGORY,
        TOOL_GET_RECURRING_PAYMENTS,
        TOOL_GET_UPCOMING_EVENTS,
        TOOL_GET_ALERTS,
        TOOL_GET_RISK_ASSESSMENT,
        TOOL_FORECAST_EXPENSES,
        TOOL_SIMULATE_PURCHASE,
        TOOL_SEARCH_DOCUMENTS,
        TOOL_GET_LOANS,
        TOOL_GET_INSURANCE,
        TOOL_GET_SUBSCRIPTIONS,
    ]
}


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return serialisable tool schemas for API introspection."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "category": t.category,
        }
        for t in TOOL_REGISTRY.values()
    ]
