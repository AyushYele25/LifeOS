"""
LIFEOS Financial Risk & Stress Testing Engine Tests

Validates:
1. Five-pillar deterministic risk breakdown (Liquidity, DTI, Cashflow, Obligations, Concentration).
2. Risk score indexing (0-100), risk levels, and risk grades (A/B/C/D/F).
3. Automated macroeconomic stress test suite (Income loss 3m/6m, +200bps rate hike, emergency shock, inflation).
4. Custom stress testing with user-specified multi-parameter shocks.
5. REST API endpoints (/api/risk/assessment, /api/risk/stress-test).
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
    RiskLevel,
)
from app.services.risk_engine import RiskEngine


@pytest.mark.asyncio
async def test_low_risk_financial_profile(db_session, test_user):
    """
    Test a financially robust profile:
    - ₹5,00,000 savings
    - ₹1,00,000 monthly salary
    - ₹35,000 monthly expenses
    - No loans
    - Expected: Low risk score, RiskLevel.LOW, Grade A, positive stress test survival.
    """
    savings = Account(
        user_id=test_user.id,
        name="HDFC Wealth",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("500000.00"),
    )
    db_session.add(savings)

    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    groc_cat = (await db_session.execute(select(Category).where(Category.name == "Groceries"))).scalar_one()
    util_cat = (await db_session.execute(select(Category).where(Category.name == "Utilities"))).scalar_one()

    today = date.today()
    for i in range(3):
        # Salary credit ₹1,00,000
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=sal_cat.id,
                date=today - timedelta(days=i * 30),
                description="Salary",
                amount=Decimal("100000.00"),
                transaction_type=TransactionType.CREDIT,
            )
        )
        # Groceries ₹20,000
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=groc_cat.id,
                date=today - timedelta(days=i * 30 + 5),
                description="Groceries",
                amount=Decimal("20000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )
        # Utilities ₹15,000
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=util_cat.id,
                date=today - timedelta(days=i * 30 + 10),
                description="Bills",
                amount=Decimal("15000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )
    await db_session.commit()

    engine = RiskEngine(db_session, test_user.id)
    assessment = await engine.get_assessment()

    assert assessment["composite_risk_score"] <= 30.0
    assert assessment["risk_level"] in [RiskLevel.LOW, RiskLevel.MEDIUM]
    assert assessment["risk_grade"] in ["A", "B"]
    assert assessment["emergency_runway_months"] >= 10.0
    assert assessment["dti_ratio"] == 0.0
    assert assessment["savings_rate"] >= 0.50

    # Verify stress tests show resilience
    stress_tests = assessment["stress_tests"]
    assert len(stress_tests) == 5

    income_loss_3m = next((s for s in stress_tests if "3 Months" in s["scenario_name"]), None)
    assert income_loss_3m is not None
    assert income_loss_3m["liquid_balance_after"] > Decimal("300000.00")
    assert income_loss_3m["is_survivable_6_months"] is True


@pytest.mark.asyncio
async def test_high_risk_stressed_profile(db_session, test_user):
    """
    Test an acutely stressed profile:
    - ₹15,000 savings
    - ₹50,000 salary
    - ₹30,000 loan EMI (DTI = 60%)
    - ₹35,000 expenses (Total outgo ₹65,000 > income ₹50,000)
    - Expected: High/Critical risk, Grade C/D/F, runway < 1 month.
    """
    savings = Account(
        user_id=test_user.id,
        name="Axis Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("15000.00"),
    )
    db_session.add(savings)

    loan = Loan(
        user_id=test_user.id,
        name="Personal Loan",
        loan_type=LoanType.PERSONAL,
        principal_amount=Decimal("800000.00"),
        emi_amount=Decimal("30000.00"),
        interest_rate=Decimal("14.5"),
        status=LoanStatus.ACTIVE,
    )
    db_session.add(loan)

    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()

    today = date.today()
    for i in range(3):
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=sal_cat.id,
                date=today - timedelta(days=i * 30),
                description="Salary",
                amount=Decimal("50000.00"),
                transaction_type=TransactionType.CREDIT,
            )
        )
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=shop_cat.id,
                date=today - timedelta(days=i * 30 + 5),
                description="Expenses",
                amount=Decimal("35000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )
    await db_session.commit()

    engine = RiskEngine(db_session, test_user.id)
    assessment = await engine.get_assessment()

    assert assessment["composite_risk_score"] >= 50.0
    assert assessment["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert assessment["dti_ratio"] >= 0.50
    assert assessment["emergency_runway_months"] < 1.0
    assert len(assessment["vulnerabilities"]) > 0
    assert len(assessment["recommendations"]) > 0


@pytest.mark.asyncio
async def test_custom_stress_test(db_session, test_user):
    """
    Test running customized stress tests simulating income drop + emergency expense + rate hike.
    """
    savings = Account(
        user_id=test_user.id,
        name="Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("200000.00"),
    )
    db_session.add(savings)

    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    today = date.today()
    for i in range(2):
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=sal_cat.id,
                date=today - timedelta(days=i * 30),
                description="Salary",
                amount=Decimal("60000.00"),
                transaction_type=TransactionType.CREDIT,
            )
        )
    await db_session.commit()

    engine = RiskEngine(db_session, test_user.id)
    custom_res = await engine.run_custom_stress_test(
        income_drop_pct=50.0,
        emergency_expense=Decimal("50000.00"),
        interest_rate_hike_bps=150,
        inflation_pct=10.0,
        duration_months=3,
    )

    assert "Custom Shock" in custom_res["scenario_name"]
    assert custom_res["liquid_balance_after"] < Decimal("200000.00")
    assert "details" in custom_res
    assert custom_res["details"]["emergency_expense_deducted"] == Decimal("50000.00")


@pytest.mark.asyncio
async def test_risk_api_endpoints(client: AsyncClient, auth_headers: dict):
    """
    Verify REST API endpoints:
    1. GET /api/risk/assessment
    2. POST /api/risk/stress-test
    """
    # 1. GET Assessment
    resp = await client.get("/api/risk/assessment", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "composite_risk_score" in data
    assert "risk_level" in data
    assert "risk_grade" in data
    assert "dti_ratio" in data
    assert "breakdown" in data
    assert len(data["breakdown"]) == 5
    assert "stress_tests" in data
    assert len(data["stress_tests"]) == 5

    # 2. POST Custom Stress Test
    stress_payload = {
        "income_drop_pct": 25.0,
        "emergency_expense": "40000.00",
        "interest_rate_hike_bps": 100,
        "inflation_pct": 8.0,
        "duration_months": 4,
    }
    st_resp = await client.post("/api/risk/stress-test", json=stress_payload, headers=auth_headers)
    assert st_resp.status_code == 200
    st_data = st_resp.json()

    assert "scenario_name" in st_data
    assert "monthly_cashflow_delta" in st_data
    assert "liquid_balance_after" in st_data
    assert "runway_months_after" in st_data
    assert "risk_level_after" in st_data
    assert "is_survivable_6_months" in st_data
