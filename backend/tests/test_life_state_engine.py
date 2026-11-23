"""
LIFEOS Life State Engine Tests

Validates:
1. Deterministic financial calculations (total balance, monthly income, monthly expenses, savings rate).
2. Emergency runway calculation (months of expenses covered).
3. Financial risk score index (0-100) and risk level assignment.
4. State comparison deltas ("What changed?").
5. Life State REST API endpoints (current, recompute, history, compare).
6. Dashboard overview integration with Life State.
"""

from datetime import date, timedelta
from decimal import Decimal
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
    Asset,
    AssetType,
    LifeState,
    RiskLevel,
)
from app.services.life_state_engine import LifeStateEngine


@pytest.mark.asyncio
async def test_recompute_state_deterministic_calculations(db_session, test_user):
    """
    Verify exact deterministic math across accounts, income, expenses, obligations, and runway.
    """
    # 1. Accounts: Savings ₹1,50,000 + Credit Card debt ₹20,000 -> Net liquid ₹1,30,000
    savings = Account(
        user_id=test_user.id,
        name="HDFC Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("150000.00"),
    )
    cc = Account(
        user_id=test_user.id,
        name="ICICI Credit Card",
        account_type=AccountType.CREDIT_CARD,
        current_balance=Decimal("20000.00"),
    )
    db_session.add_all([savings, cc])

    # 2. Income & Expenses: 3 months of ₹80,000 salary and ₹40,000 expenses
    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()

    today = date.today()
    for i in range(3):
        # Salary credit
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=sal_cat.id,
                date=today - timedelta(days=i * 30),
                description="Monthly Salary",
                amount=Decimal("80000.00"),
                transaction_type=TransactionType.CREDIT,
            )
        )
        # Expense debit
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=shop_cat.id,
                date=today - timedelta(days=i * 30 + 5),
                description="Household Spending",
                amount=Decimal("40000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )

    # 3. Active Loan: EMI ₹10,000
    loan = Loan(
        user_id=test_user.id,
        name="Auto Loan",
        loan_type=LoanType.VEHICLE,
        principal_amount=Decimal("500000.00"),
        emi_amount=Decimal("10000.00"),
        status=LoanStatus.ACTIVE,
    )
    # 4. Asset: ₹6,00,000 car
    car = Asset(
        user_id=test_user.id,
        name="Hyundai Creta",
        asset_type=AssetType.VEHICLE,
        current_value=Decimal("600000.00"),
        is_active=True,
    )
    db_session.add_all([loan, car])
    await db_session.commit()

    engine = LifeStateEngine(db_session, test_user.id)
    state = await engine.recompute_state(save_snapshot=True)

    assert state.total_balance == Decimal("130000.00")
    assert state.monthly_income == Decimal("80000.00")
    assert state.monthly_expenses == Decimal("40000.00")
    assert state.monthly_savings == Decimal("40000.00")
    assert state.savings_rate == 0.50
    assert state.total_emi_burden == Decimal("10000.00")
    assert state.total_asset_value == Decimal("600000.00")
    assert state.emergency_runway_months == 3.25
    assert state.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    assert 0.0 <= state.financial_risk_score <= 50.0


@pytest.mark.asyncio
async def test_critical_risk_score_on_high_dti_and_low_runway(db_session, test_user):
    """
    Verify high debt (DTI 70%), low runway (< 0.2 months), and negative savings trigger CRITICAL risk.
    """
    # Liquid balance only ₹10,000
    acc = Account(
        user_id=test_user.id,
        name="Checking",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("10000.00"),
    )
    db_session.add(acc)

    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()

    today = date.today()
    # Income ₹50,000, Expenses ₹55,000
    db_session.add(
        Transaction(
            user_id=test_user.id,
            account_id=acc.id,
            category_id=sal_cat.id,
            date=today,
            description="Salary",
            amount=Decimal("50000.00"),
            transaction_type=TransactionType.CREDIT,
        )
    )
    db_session.add(
        Transaction(
            user_id=test_user.id,
            account_id=acc.id,
            category_id=shop_cat.id,
            date=today,
            description="Expenses",
            amount=Decimal("55000.00"),
            transaction_type=TransactionType.DEBIT,
        )
    )

    # Debt: EMI ₹35,000 (DTI = 70%)
    loan = Loan(
        user_id=test_user.id,
        name="Personal Loan",
        loan_type=LoanType.PERSONAL,
        principal_amount=Decimal("1000000.00"),
        emi_amount=Decimal("35000.00"),
        status=LoanStatus.ACTIVE,
    )
    db_session.add(loan)
    await db_session.commit()

    engine = LifeStateEngine(db_session, test_user.id)
    state = await engine.recompute_state(save_snapshot=True)

    assert state.emergency_runway_months < 1.0
    assert state.savings_rate < 0.0
    assert state.financial_risk_score >= 70.0
    assert state.risk_level == RiskLevel.CRITICAL
    assert "debt_burden" in state.risk_breakdown
    assert "runway" in state.risk_breakdown


@pytest.mark.asyncio
async def test_state_comparison_deltas(db_session, test_user):
    """
    Verify successive recomputations identify financial shifts between snapshots.
    """
    acc = Account(
        user_id=test_user.id,
        name="Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("100000.00"),
    )
    db_session.add(acc)
    await db_session.commit()

    engine = LifeStateEngine(db_session, test_user.id)
    state1 = await engine.recompute_state(save_snapshot=True)

    # Change balance by ₹50,000
    acc.current_balance = Decimal("150000.00")
    await db_session.commit()

    state2 = await engine.recompute_state(save_snapshot=True)

    assert state2.changes_from_previous is not None
    deltas = state2.changes_from_previous
    bal_change = next((d for d in deltas if d["field"] == "total_balance"), None)
    assert bal_change is not None
    assert bal_change["old"] == 100000.0
    assert bal_change["new"] == 150000.0
    assert bal_change["change"] == 50000.0
    assert bal_change["change_pct"] == 50.0


@pytest.mark.asyncio
async def test_lifestate_api_and_dashboard_integration(
    client: AsyncClient, test_user: User, auth_headers: dict, db_session
):
    """
    Tests REST API endpoints:
    1. POST /api/life-state/recompute
    2. GET /api/life-state/current
    3. GET /api/life-state/history
    4. GET /api/life-state/compare
    5. GET /api/dashboard/overview (confirms dashboard includes life state)
    """
    # 1. Force recompute via API
    recompute_resp = await client.post("/api/life-state/recompute", headers=auth_headers)
    assert recompute_resp.status_code == 200
    state_data = recompute_resp.json()
    assert "financial_risk_score" in state_data
    assert "emergency_runway_months" in state_data

    # 2. Get current state
    curr_resp = await client.get("/api/life-state/current", headers=auth_headers)
    assert curr_resp.status_code == 200
    assert curr_resp.json()["id"] == state_data["id"]

    # 3. Get history
    hist_resp = await client.get("/api/life-state/history", headers=auth_headers)
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()) >= 1

    # 4. Compare states
    comp_resp = await client.get("/api/life-state/compare", headers=auth_headers)
    assert comp_resp.status_code == 200
    assert "current_state" in comp_resp.json()

    # 5. Dashboard overview
    dash_resp = await client.get("/api/dashboard/overview", headers=auth_headers)
    assert dash_resp.status_code == 200
    dash_data = dash_resp.json()
    assert dash_data["life_state"] is not None
    assert dash_data["life_state"]["id"] == state_data["id"]
