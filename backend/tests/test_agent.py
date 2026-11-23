"""
LIFEOS Agentic AI Tests

Validates:
1. Tool registry completeness (13 typed tools registered).
2. Individual tool invocation correctness (get_life_state, get_transactions, etc.).
3. Intent classification and multi-tool routing.
4. Parameter extraction from natural language.
5. Multi-tool chaining and follow-up tool selection.
6. Read-only safety guardrails (no writes via agent).
7. Evidence grounding — all numbers in response trace to tool results.
8. REST API endpoint tests (POST /api/agent/chat, GET /api/agent/tools).
"""

from datetime import date, timedelta
from decimal import Decimal
import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.models import (
    User,
    Account,
    AccountType,
    Transaction,
    TransactionType,
    Category,
    Loan,
    LoanType,
    LoanStatus,
    Insurance,
    InsuranceType,
    Subscription,
    SubscriptionStatus,
    SubscriptionFrequency,
    Event,
    EventType,
    EventStatus,
    RecurringPattern,
    Alert,
    AlertType,
    AlertSeverity,
    Simulation,
    RiskLevel,
)
from app.services.agent_tools import TOOL_REGISTRY, get_tool_schemas
from app.services.agent_service import AgentService
from app.services.auth_service import create_access_token


# ─── Helper: seed a financially complete test user ────

async def _seed_full_user(db_session, user):
    """Populate accounts, transactions, loans, insurance, and subscriptions for testing."""
    # Account
    savings = Account(
        user_id=user.id,
        name="HDFC Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("250000.00"),
    )
    db_session.add(savings)

    # Categories
    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    food_cat = (await db_session.execute(select(Category).where(Category.name == "Food & Dining"))).scalar_one()
    rent_cat = (await db_session.execute(select(Category).where(Category.name == "Rent"))).scalar_one()
    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()
    emi_cat = (await db_session.execute(select(Category).where(Category.name == "EMI & Loans"))).scalar_one()

    today = date.today()
    # Salary credits — 3 months
    for i in range(3):
        db_session.add(Transaction(
            user_id=user.id, account_id=savings.id, category_id=sal_cat.id,
            date=today - timedelta(days=i * 30),
            description="Salary Credit", amount=Decimal("85000.00"),
            transaction_type=TransactionType.CREDIT,
        ))

    # Debits — groceries, rent, food
    debits = [
        (food_cat.id, "Swiggy Order", "1250.00"),
        (food_cat.id, "Zomato Order", "890.00"),
        (rent_cat.id, "House Rent", "18000.00"),
        (shop_cat.id, "Amazon Purchase", "4500.00"),
        (emi_cat.id, "Car Loan EMI", "12500.00"),
    ]
    for cat_id, desc, amt in debits:
        for i in range(2):
            db_session.add(Transaction(
                user_id=user.id, account_id=savings.id, category_id=cat_id,
                date=today - timedelta(days=i * 30 + 3),
                description=desc, amount=Decimal(amt),
                transaction_type=TransactionType.DEBIT,
            ))

    # Loan
    loan = Loan(
        user_id=user.id,
        name="Car Loan",
        loan_type=LoanType.VEHICLE,
        lender="HDFC Bank",
        principal_amount=Decimal("500000.00"),
        outstanding_amount=Decimal("380000.00"),
        interest_rate=9.5,
        emi_amount=Decimal("12500.00"),
        emi_day=5,
        total_emis=48,
        emis_paid=10,
        status=LoanStatus.ACTIVE,
    )
    db_session.add(loan)

    # Insurance
    insurance = Insurance(
        user_id=user.id,
        name="Health Insurance",
        insurance_type=InsuranceType.HEALTH,
        provider="Star Health",
        policy_number="SH-2024-001",
        premium_amount=Decimal("18000.00"),
        premium_frequency=SubscriptionFrequency.YEARLY,
        sum_insured=Decimal("500000.00"),
        expiry_date=today + timedelta(days=120),
    )
    db_session.add(insurance)

    # Subscription
    sub = Subscription(
        user_id=user.id,
        name="Netflix",
        provider="Netflix",
        amount=Decimal("649.00"),
        frequency=SubscriptionFrequency.MONTHLY,
        next_billing_date=today + timedelta(days=15),
        status=SubscriptionStatus.ACTIVE,
        category="entertainment",
    )
    db_session.add(sub)

    # Recurring pattern
    pattern = RecurringPattern(
        user_id=user.id,
        merchant="Netflix",
        category="Subscriptions",
        average_amount=Decimal("649.00"),
        frequency=SubscriptionFrequency.MONTHLY,
        confidence=0.92,
        transaction_count=6,
        next_expected_date=today + timedelta(days=15),
    )
    db_session.add(pattern)

    # Upcoming event
    event = Event(
        user_id=user.id,
        event_type=EventType.LOAN_EMI,
        title="Car Loan EMI Due",
        event_date=today + timedelta(days=5),
        amount=Decimal("12500.00"),
        status=EventStatus.UPCOMING,
        source_type="loan",
    )
    db_session.add(event)

    # Alert
    alert = Alert(
        user_id=user.id,
        alert_type=AlertType.ANOMALY,
        severity=AlertSeverity.WARNING,
        title="Unusual spending at Amazon",
        message="₹4,500 spend at Amazon is 180% above your baseline for Shopping.",
    )
    db_session.add(alert)

    await db_session.commit()
    return savings


# ═══════════════════════════════════════════════════════
# Test 1: Tool Registry Completeness
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tool_registry_completeness():
    """Verify all 13 tools are registered with correct metadata."""
    expected_tools = {
        "get_life_state", "get_transactions", "get_spending_by_category",
        "get_recurring_payments", "get_upcoming_events", "get_alerts",
        "get_risk_assessment", "forecast_expenses", "simulate_purchase",
        "search_documents", "get_loans", "get_insurance", "get_subscriptions",
    }
    registered = set(TOOL_REGISTRY.keys())
    assert registered == expected_tools, f"Missing: {expected_tools - registered}, Extra: {registered - expected_tools}"
    assert len(TOOL_REGISTRY) == 13

    # Each tool must have name, description, parameters, and execute
    for name, tool in TOOL_REGISTRY.items():
        assert tool.name == name
        assert len(tool.description) > 20, f"Tool {name} description too short"
        assert "type" in tool.parameters
        assert callable(tool.execute)

    # get_tool_schemas should return serialisable dicts
    schemas = get_tool_schemas()
    assert len(schemas) == 13
    for s in schemas:
        assert "name" in s
        assert "description" in s
        assert "parameters" in s
        assert "category" in s


# ═══════════════════════════════════════════════════════
# Test 2: Life State Tool Execution
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tool_get_life_state(db_session, test_user):
    """Tool: get_life_state returns a complete financial snapshot."""
    await _seed_full_user(db_session, test_user)

    result = await TOOL_REGISTRY["get_life_state"].execute(db_session, test_user.id, {})

    assert "total_balance" in result
    assert "monthly_income" in result
    assert "monthly_expenses" in result
    assert "savings_rate" in result
    assert "emergency_runway_months" in result
    assert "financial_risk_score" in result
    assert "risk_level" in result
    assert result["total_balance"] > 0


# ═══════════════════════════════════════════════════════
# Test 3: Intent Classification & Multi-Tool Routing
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_intent_classification_and_routing(db_session, test_user):
    """Agent correctly classifies intents and routes to appropriate tools."""
    agent = AgentService(db_session, test_user.id)

    # Affordability → simulate_purchase + get_life_state
    tools = agent._select_tools("Can I afford a laptop for ₹80,000?")
    tool_names = [t.name for t in tools]
    assert "simulate_purchase" in tool_names
    assert "get_life_state" in tool_names

    # Spending → get_spending_by_category
    tools = agent._select_tools("Where is my money going?")
    tool_names = [t.name for t in tools]
    assert "get_spending_by_category" in tool_names

    # Risk → get_risk_assessment
    tools = agent._select_tools("How is my financial health?")
    tool_names = [t.name for t in tools]
    assert "get_risk_assessment" in tool_names

    # Upcoming → get_upcoming_events
    tools = agent._select_tools("What am I forgetting?")
    tool_names = [t.name for t in tools]
    assert "get_upcoming_events" in tool_names

    # Fallback → get_life_state
    tools = agent._select_tools("Tell me something interesting")
    tool_names = [t.name for t in tools]
    assert "get_life_state" in tool_names


# ═══════════════════════════════════════════════════════
# Test 4: Parameter Extraction
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_parameter_extraction(db_session, test_user):
    """Agent extracts typed parameters from natural language."""
    agent = AgentService(db_session, test_user.id)

    # Purchase params — amount + item
    params = agent._extract_purchase_params("Can I afford a laptop for ₹80000?")
    assert params["amount"] == 80000.0
    assert "laptop" in params["item"].lower()

    # Spending params — time period
    params = agent._extract_spending_params("Show spending for last 3 months")
    assert params["days"] == 90

    # Event params — horizon
    params = agent._extract_event_params("What's due next month?")
    assert params["horizon_days"] == 30


# ═══════════════════════════════════════════════════════
# Test 5: Full Agent Chat — Grounded Response
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_chat_grounded_response(db_session, test_user):
    """
    End-to-end agent chat: user asks about financial state,
    agent calls get_life_state, returns grounded response with actual numbers.
    """
    await _seed_full_user(db_session, test_user)

    agent = AgentService(db_session, test_user.id)
    result = await agent.chat("What's my financial overview?")

    # Must return structured response
    assert "conversation_id" in result
    assert "message" in result
    assert "tool_calls" in result
    assert "tools_used" in result
    assert "data" in result

    # Must have used get_life_state
    assert "get_life_state" in result["tools_used"]

    # Message should contain actual financial data markers (grounded)
    msg = result["message"]
    assert "₹" in msg, "Response should contain rupee amounts from tool data"
    assert "Balance" in msg or "balance" in msg.lower()

    # tool_calls should be non-empty
    assert len(result["tool_calls"]) >= 1
    assert result["tool_calls"][0]["tool_name"] == "get_life_state"


# ═══════════════════════════════════════════════════════
# Test 6: Multi-Tool Chaining
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_multi_tool_chaining(db_session, test_user):
    """Agent chains multiple tools for compound queries like 'can I afford X?'."""
    await _seed_full_user(db_session, test_user)

    agent = AgentService(db_session, test_user.id)
    result = await agent.chat("Can I afford to buy a phone for ₹50000?")

    # Should call simulate_purchase + get_life_state
    assert len(result["tools_used"]) >= 2
    assert "simulate_purchase" in result["tools_used"]
    assert "get_life_state" in result["tools_used"]

    # Should have scenarios in the simulation result
    sim_data = result["data"].get("simulate_purchase", {})
    assert "scenarios" in sim_data
    assert "recommendation" in sim_data


# ═══════════════════════════════════════════════════════
# Test 7: Read-Only Safety — No Mutations
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_read_only_safety_guardrails(db_session, test_user):
    """
    Agent tools are strictly read-only.
    Verify that running agent queries doesn't create any new database records.
    """
    await _seed_full_user(db_session, test_user)

    # Count records before
    from sqlalchemy import func
    txn_count_before = (await db_session.execute(
        select(func.count(Transaction.id)).where(Transaction.user_id == test_user.id)
    )).scalar()
    sim_count_before = (await db_session.execute(
        select(func.count(Simulation.id))
    )).scalar()

    agent = AgentService(db_session, test_user.id)
    await agent.chat("Can I afford a laptop for ₹80000?")
    await agent.chat("Show my transactions")
    await agent.chat("What's my risk assessment?")

    # Count records after — should be identical
    txn_count_after = (await db_session.execute(
        select(func.count(Transaction.id)).where(Transaction.user_id == test_user.id)
    )).scalar()
    sim_count_after = (await db_session.execute(
        select(func.count(Simulation.id))
    )).scalar()

    assert txn_count_before == txn_count_after, "Agent should not create transactions"
    assert sim_count_before == sim_count_after, "Agent should not persist simulations"


# ═══════════════════════════════════════════════════════
# Test 8: REST API — POST /api/agent/chat
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_chat_api_endpoint(client, db_session, test_user, auth_headers):
    """POST /api/agent/chat returns a grounded response via the REST API."""
    await _seed_full_user(db_session, test_user)

    resp = await client.post(
        "/api/agent/chat",
        json={"message": "What's my financial overview?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    body = resp.json()
    assert "conversation_id" in body
    assert "message" in body
    assert "tool_calls" in body
    assert "tools_used" in body
    assert len(body["message"]) > 50, "Response should be substantial"
    assert "get_life_state" in body["tools_used"]


# ═══════════════════════════════════════════════════════
# Test 9: REST API — GET /api/agent/tools
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_tools_introspection_endpoint(client, auth_headers):
    """GET /api/agent/tools returns all 13 tool schemas."""
    resp = await client.get("/api/agent/tools", headers=auth_headers)
    assert resp.status_code == 200

    tools = resp.json()
    assert len(tools) == 13

    tool_names = {t["name"] for t in tools}
    assert "get_life_state" in tool_names
    assert "simulate_purchase" in tool_names
    assert "search_documents" in tool_names

    # Each tool should have description and parameters
    for t in tools:
        assert len(t["description"]) > 20
        assert "type" in t["parameters"]


# ═══════════════════════════════════════════════════════
# Test 10: Max Tool Call Limit
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_max_tool_call_limit(db_session, test_user):
    """Agent respects the MAX_TOOL_CALLS_PER_QUERY limit (5)."""
    await _seed_full_user(db_session, test_user)

    agent = AgentService(db_session, test_user.id)
    # A query that could match many tools
    result = await agent.chat(
        "Show me my balance, spending, transactions, recurring payments, "
        "subscriptions, loans, insurance, and upcoming events"
    )

    assert len(result["tool_calls"]) <= 5, "Agent must not exceed 5 tool calls"
