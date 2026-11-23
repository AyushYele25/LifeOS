"""
LIFEOS Decision Simulator Tests

Validates:
1. "Can I afford X?" purchase simulation (Cash upfront vs EMI installments vs Delay & Save Up).
2. Loan tenure comparison (Standard tenure vs Extended tenure with interest delta).
3. Salary adjustments and new recurring expenses.
4. Deterministic recommendations and risk level transitions.
5. Database persistence and REST API endpoints (/api/simulations).
6. Multi-user isolation enforcement on simulation records.
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
    RiskLevel,
    Simulation,
)
from app.services.scenario_simulator import ScenarioSimulator
from app.services.auth_service import create_access_token


@pytest.mark.asyncio
async def test_simulate_discretionary_purchase(db_session, test_user):
    """
    Test purchase simulation with sufficient liquidity:
    - User balance ₹2,50,000, monthly expenses ₹40,000, salary ₹90,000
    - Item: ₹1,00,000 Laptop
    - Evaluates Cash vs 12m EMI vs Delay
    """
    savings = Account(
        user_id=test_user.id,
        name="HDFC Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("250000.00"),
    )
    db_session.add(savings)

    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()

    today = date.today()
    for i in range(2):
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=savings.id,
                category_id=sal_cat.id,
                date=today - timedelta(days=i * 30),
                description="Salary",
                amount=Decimal("90000.00"),
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
                amount=Decimal("40000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )
    await db_session.commit()

    engine = ScenarioSimulator(db_session, test_user.id)
    sim = await engine.run_simulation(
        simulation_type="purchase",
        title="Can I buy MacBook Pro M3?",
        params={"item": "MacBook Pro", "amount": 100000, "emi_months": 12, "emi_interest_rate": 14.0},
    )

    assert sim.id is not None
    assert sim.title == "Can I buy MacBook Pro M3?"
    assert len(sim.scenarios) == 3

    cash_path = next(s for s in sim.scenarios if "Cash" in s["name"])
    assert Decimal(str(cash_path["savings_after"])) == Decimal("150000.00")
    assert cash_path["emergency_runway_months"] >= 3.0

    emi_path = next(s for s in sim.scenarios if "EMI" in s["name"])
    assert Decimal(str(emi_path["savings_after"])) == Decimal("250000.00")
    assert Decimal(str(emi_path["monthly_cashflow_impact"])) < Decimal("0.00")
    assert Decimal(str(emi_path["details"]["total_interest_paid"])) > Decimal("0.00")

    assert "RECOMMENDED" in sim.recommendation


@pytest.mark.asyncio
async def test_simulate_purchase_under_liquidity_stress(db_session, test_user):
    """
    Test purchase when liquid balance is insufficient:
    - User balance ₹30,000, monthly expenses ₹35,000
    - Item: ₹50,000 Smartphone
    - Cash path should drive balance negative and trigger CRITICAL risk.
    """
    savings = Account(
        user_id=test_user.id,
        name="Axis Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("30000.00"),
    )
    db_session.add(savings)

    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    db_session.add(
        Transaction(
            user_id=test_user.id,
            account_id=savings.id,
            category_id=sal_cat.id,
            date=date.today(),
            description="Salary",
            amount=Decimal("40000.00"),
            transaction_type=TransactionType.CREDIT,
        )
    )
    await db_session.commit()

    engine = ScenarioSimulator(db_session, test_user.id)
    sim = await engine.run_simulation(
        simulation_type="purchase",
        title="Can I buy iPhone 15?",
        params={"item": "iPhone 15", "amount": 50000, "emi_months": 6},
    )

    cash_path = next(s for s in sim.scenarios if "Cash" in s["name"])
    assert Decimal(str(cash_path["savings_after"])) == Decimal("-20000.00")
    assert cash_path["risk_level_after"] in [RiskLevel.CRITICAL, "CRITICAL", "critical"]
    assert "NOT RECOMMENDED" in sim.recommendation or "Delay" in sim.recommendation


@pytest.mark.asyncio
async def test_simulate_loan_and_salary_change(db_session, test_user):
    """
    Test loan tenure comparison and compensation adjustment scenarios.
    """
    savings = Account(
        user_id=test_user.id,
        name="Primary",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("150000.00"),
    )
    db_session.add(savings)
    sal_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()
    db_session.add(
        Transaction(
            user_id=test_user.id,
            account_id=savings.id,
            category_id=sal_cat.id,
            date=date.today(),
            description="Salary",
            amount=Decimal("80000.00"),
            transaction_type=TransactionType.CREDIT,
        )
    )
    await db_session.commit()

    engine = ScenarioSimulator(db_session, test_user.id)

    # 1. Loan simulation
    loan_sim = await engine.run_simulation(
        simulation_type="loan",
        title="New Car Loan ₹6 Lakhs",
        params={"principal": 600000, "tenure_months": 36, "interest_rate": 9.5},
    )
    assert len(loan_sim.scenarios) == 2
    std_scen = loan_sim.scenarios[0]
    ext_scen = loan_sim.scenarios[1]
    # Extended tenure has lower monthly payment but higher total interest
    assert abs(ext_scen["monthly_cashflow_impact"]) < abs(std_scen["monthly_cashflow_impact"])
    assert ext_scen["details"]["total_interest"] > std_scen["details"]["total_interest"]

    # 2. Salary change simulation
    sal_sim = await engine.run_simulation(
        simulation_type="salary_change",
        title="Promotion to Senior Role",
        params={"new_salary": 110000},
    )
    assert Decimal(str(sal_sim.scenarios[0]["monthly_cashflow_impact"])) == Decimal("30000.00")


@pytest.mark.asyncio
async def test_simulations_api_lifecycle_and_isolation(client: AsyncClient, auth_headers: dict, db_session):
    """
    Verify API endpoints:
    1. POST /api/simulations creates and persists simulation
    2. GET /api/simulations retrieves user's list
    3. GET /api/simulations/{id} retrieves detail
    4. Per-user isolation prevents unauthorized access
    """
    # 1. POST create simulation
    payload = {
        "simulation_type": "purchase",
        "title": "Can I buy 65-inch OLED TV?",
        "params": {"item": "OLED TV", "amount": 120000, "emi_months": 12},
    }
    resp = await client.post("/api/simulations", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    sim_data = resp.json()
    sim_id = sim_data["id"]
    assert sim_data["title"] == "Can I buy 65-inch OLED TV?"
    assert len(sim_data["scenarios"]) == 3
    assert sim_data["recommendation"] is not None

    # 2. GET list
    list_resp = await client.get("/api/simulations", headers=auth_headers)
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert any(s["id"] == sim_id for s in items)

    # 3. GET detail
    detail_resp = await client.get(f"/api/simulations/{sim_id}", headers=auth_headers)
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == sim_id

    # 4. Multi-user isolation test: create a second user
    other_user = User(
        email="other_sim_user@example.com",
        full_name="Other User",
        hashed_password="fakehash",
        is_active=True,
    )
    db_session.add(other_user)
    await db_session.commit()
    await db_session.refresh(other_user)

    other_token = create_access_token(user_id=other_user.id, email=other_user.email)
    other_headers = {"Authorization": f"Bearer {other_token}"}

    # Other user must NOT be able to view first user's simulation (404)
    iso_resp = await client.get(f"/api/simulations/{sim_id}", headers=other_headers)
    assert iso_resp.status_code == 404
