"""
Unit tests for the 6 Innovation Pillars:
1. Quick Ingest (Natural language voice/text & receipt OCR)
2. Daily Safe-to-Spend Engine
3. Autonomous Financial Defense (Zombies & Debt Snowball/Avalanche)
4. Indian Tax Copilot (Old vs New Regime)
5. Household Shared Vault & Split Ledger
"""

import pytest
from httpx import AsyncClient
from datetime import date, timedelta
from decimal import Decimal

from app.models.models import (
    Loan, LoanType, LoanStatus,
    Subscription, SubscriptionFrequency, SubscriptionStatus,
    Goal, GoalStatus, Account, AccountType,
)
from app.services.life_state_engine import LifeStateEngine
from app.services.quick_ingest import parse_natural_language_expense, parse_receipt_text
from app.services.tax_engine import compare_tax_regimes
from app.services.financial_defense import get_credit_card_recommendation


def test_quick_natural_language_expense_parsing():
    """Pillar 1: Test parsing conversational voice/text strings."""
    res1 = parse_natural_language_expense("Spent 450 at CCD on coffee via HDFC")
    assert res1["amount"] == 450.0
    assert res1["transaction_type"] == "debit"
    assert "Food" in res1["predicted_category"] or "Dining" in res1["predicted_category"]

    res2 = parse_natural_language_expense("Petrol 2200 at BPCL pump")
    assert res2["amount"] == 2200.0
    assert res2["transaction_type"] == "debit"
    assert res2["predicted_category"] == "Fuel"

    res3 = parse_natural_language_expense("Infosys monthly salary 120000 credited")
    assert res3["amount"] == 120000.0
    assert res3["transaction_type"] == "credit"
    assert res3["predicted_category"] == "Salary"


def test_receipt_ocr_parsing():
    """Pillar 1: Test receipt text extraction."""
    sample_bill = """
    STARBUCKS COFFEE INDIA
    Indiranagar Bangalore
    Date: 24/11/2026
    1x Caramel Macchiato  350.00
    1x Chocolate Croissant 280.00
    CGST 2.5%             15.75
    SGST 2.5%             15.75
    Grand Total: ₹661.50
    """
    res = parse_receipt_text(sample_bill)
    assert res["total_amount"] == 661.50
    assert "starbucks" in res["merchant"].lower()
    assert "Food" in res["predicted_category"] or "Dining" in res["predicted_category"]


def test_tax_engine_regime_comparison():
    """Pillar 5: Test Indian income tax calculations for Old vs New Regime."""
    res = compare_tax_regimes(
        gross_income=1500000.0,
        hra_exemption=180000.0,
        sec_80c=150000.0,
        sec_80d=25000.0,
        sec_80ccd_nps=50000.0,
    )
    assert res["gross_income"] == 1500000.0
    assert res["new_regime"]["total_tax_payable"] > 0
    assert res["old_regime"]["total_tax_payable"] > 0
    assert "winner" in res
    assert res["annual_tax_savings"] >= 0


def test_credit_card_rewards_optimizer():
    """Pillar 4: Test credit card recommendation for merchant categories."""
    swiggy_res = get_credit_card_recommendation("Food & Dining", 850.0)
    assert "HDFC" in swiggy_res["recommended_card"]
    assert swiggy_res["estimated_cashback"] > 0

    amazon_res = get_credit_card_recommendation("Shopping", 5000.0)
    assert "ICICI Amazon" in amazon_res["recommended_card"]
    assert amazon_res["estimated_cashback"] > 0


@pytest.mark.asyncio
async def test_pillars_api_endpoints(client: AsyncClient, auth_headers: dict, db_session, test_user):
    """Integration test for API routes under /api/pillars with active LifeState, Loans, Subs & Goals."""
    today = date.today()
    db_session.add(Account(
        user_id=test_user.id,
        name="Primary HDFC",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("250000.00"),
    ))
    db_session.add(Loan(
        user_id=test_user.id,
        name="Car Loan",
        loan_type=LoanType.VEHICLE,
        principal_amount=Decimal("500000.00"),
        outstanding_amount=Decimal("320000.00"),
        interest_rate=9.2,
        emi_amount=Decimal("14000.00"),
        total_emis=36,
        emis_paid=12,
        status=LoanStatus.ACTIVE,
    ))
    db_session.add(Subscription(
        user_id=test_user.id,
        name="Netflix Premium",
        provider="Netflix",
        amount=Decimal("649.00"),
        frequency=SubscriptionFrequency.MONTHLY,
        next_billing_date=today + timedelta(days=10),
        status=SubscriptionStatus.ACTIVE,
    ))
    db_session.add(Goal(
        user_id=test_user.id,
        name="Emergency Fund",
        target_amount=Decimal("300000.00"),
        current_amount=Decimal("150000.00"),
        status=GoalStatus.ACTIVE,
    ))
    await db_session.commit()

    lse = LifeStateEngine(db_session, test_user.id)
    await lse.recompute_state()

    # 1. Safe to Spend (with populated LifeState, Loan, Subscription, Goal)
    s2s_res = await client.get("/api/pillars/safe-to-spend", headers=auth_headers)
    assert s2s_res.status_code == 200
    s2s_data = s2s_res.json()
    assert s2s_data["daily_safe_to_spend"] > 0
    assert s2s_data["discretionary_pool_total"] > 0

    # 2. Defense - Zombies
    z_res = await client.get("/api/pillars/defense/zombies", headers=auth_headers)
    assert z_res.status_code == 200
    z_data = z_res.json()
    assert z_data["total_active_subscriptions"] == 1
    assert z_data["subscriptions"][0]["merchant"] == "Netflix Premium"

    # 3. Defense - Debt Payoff
    d_res = await client.get("/api/pillars/defense/debt-payoff?extra_monthly=3000", headers=auth_headers)
    assert d_res.status_code == 200
    d_data = d_res.json()
    assert d_data["has_debt"] is True
    assert d_data["months_saved"] >= 1

    # 4. Tax Compare Endpoint
    tax_payload = {
        "gross_annual_income": 1200000.0,
        "hra_exemption": 120000.0,
        "sec_80c": 150000.0,
        "sec_80d": 25000.0,
        "sec_80ccd_nps": 50000.0,
    }
    tax_res = await client.post("/api/pillars/tax/compare", json=tax_payload)
    assert tax_res.status_code == 200
    tax_data = tax_res.json()
    assert tax_data["gross_income"] == 1200000.0
    assert "winner" in tax_data

    # 5. Quick Ingest Parsing
    ingest_res = await client.post("/api/pillars/ingest/parse", json={"text": "Zomato dinner 520 via UPI"})
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["amount"] == 520.0
    assert ingest_data["predicted_category"] == "Food & Dining"

    # 6. Household Vault
    vault_res = await client.get("/api/pillars/household/vault", headers=auth_headers)
    assert vault_res.status_code == 200
    vault_data = vault_res.json()
    assert len(vault_data["members"]) >= 2
    assert "settlements" in vault_data


def test_sms_and_spacing_ingest_parsing():
    """Verify Bank/UPI SMS parsing and clean spacing in quick ingest."""
    res_voice = parse_natural_language_expense("Spent 450 on Swiggy dinner")
    assert res_voice["amount"] == 450.0
    assert "Spenton" not in res_voice["description"]
    assert res_voice["predicted_category"] == "Food & Dining"

    res_sms = parse_natural_language_expense("Rs.450.00 debited from A/c XX4012 on 26-11-26 to ZOMATO (UPI Ref 6331) Avl Bal: Rs.12,000.00")
    assert res_sms["amount"] == 450.0
    assert res_sms["transaction_type"] == "debit"
    assert "zomato" in res_sms["description"].lower()
    assert res_sms["predicted_category"] == "Food & Dining"


@pytest.mark.asyncio
async def test_real_user_onboarding_crud_and_vault_isolation(client: AsyncClient, auth_headers: dict):
    """Verify empty user safe-to-spend honesty, quick-setup, tx delete balance reversal, and per-user vault."""
    # 1. Empty user should get needs_setup=True instead of fake 1.2L income
    empty_s2s = await client.get("/api/pillars/safe-to-spend", headers=auth_headers)
    assert empty_s2s.status_code == 200
    assert empty_s2s.json()["needs_setup"] is True
    assert empty_s2s.json()["monthly_income"] == 0.0

    # 2. Quick setup computes real daily allowance from user inputs
    setup_res = await client.post(
        "/api/pillars/quick-setup",
        json={
            "liquid_balance": 25000.0,
            "monthly_income": 35000.0,
            "fixed_monthly_bills": 10000.0,
            "account_name": "Student SBI",
        },
        headers=auth_headers,
    )
    assert setup_res.status_code == 200
    s2s_after = setup_res.json()["safe_to_spend"]
    assert s2s_after["needs_setup"] is False
    assert s2s_after["monthly_income"] == 35000.0
    assert s2s_after["daily_safe_to_spend"] > 0

    # 3. Create & delete transaction reverses account balance
    acc_before = (await client.get("/api/accounts", headers=auth_headers)).json()[0]
    tx_res = await client.post(
        "/api/transactions",
        json={"date": str(date.today()), "description": "Swiggy order", "amount": 500.0, "transaction_type": "debit"},
        headers=auth_headers,
    )
    assert tx_res.status_code == 201
    tx_id = tx_res.json()["id"]
    acc_mid = (await client.get("/api/accounts", headers=auth_headers)).json()[0]
    assert float(acc_mid["current_balance"]) == float(acc_before["current_balance"]) - 500.0

    del_res = await client.delete(f"/api/transactions/{tx_id}", headers=auth_headers)
    assert del_res.status_code == 204
    acc_restored = (await client.get("/api/accounts", headers=auth_headers)).json()[0]
    assert float(acc_restored["current_balance"]) == float(acc_before["current_balance"])

    # 4. Customize Shared Vault members & add/delete bill privately
    mem_res = await client.put(
        "/api/pillars/household/members",
        json={"vault_name": "Hostel Room 12", "members": ["Arjun", "Vikram"]},
        headers=auth_headers,
    )
    assert mem_res.status_code == 200
    assert mem_res.json()["vault_name"] == "Hostel Room 12"
    split_res = await client.post(
        "/api/pillars/household/split",
        json={"description": "Wifi Bill", "amount": 1000.0, "paid_by": "Arjun"},
        headers=auth_headers,
    )
    assert split_res.status_code == 200
    assert split_res.json()["total_shared_spend"] == 1000.0
    exp_id = split_res.json()["expenses"][0]["id"]
    del_exp = await client.delete(f"/api/pillars/household/expense/{exp_id}", headers=auth_headers)
    assert del_exp.status_code == 200
    assert del_exp.json()["total_shared_spend"] == 0.0

    # 5. Agent chat builds actionable monthly game plan
    chat_res = await client.post(
        "/api/agent/chat",
        json={"message": "How can I save more money and build a monthly budget plan?"},
        headers=auth_headers,
    )
    assert chat_res.status_code == 200
    assert "Actionable Monthly Game Plan" in chat_res.json()["message"]


@pytest.mark.asyncio
async def test_small_bug_regressions(client: AsyncClient, auth_headers: dict, db_session, test_user):
    """Verify fixes for zero-prepay debt, Budget 2025 marginal relief, everyday ML rules, and purchase intent routing."""
    from app.ml.classifier import predict_category
    from app.services.tax_engine import calculate_new_regime_tax

    # 1. Tax Engine FY 2025-26/2026-27 rebate & marginal relief (no cliff)
    t_zero = calculate_new_regime_tax(1275000.0)
    assert t_zero["total_tax_payable"] == 0.0
    t_marginal = calculate_new_regime_tax(1276000.0)
    assert 0 < t_marginal["total_tax_payable"] <= 1040.0

    # 2. Everyday ML rules & quick ingest cleanup
    assert predict_category("Tea 20")["predicted_category"] == "Food & Dining"
    assert predict_category("Paid 450 to Rahul for dinner")["predicted_category"] == "Food & Dining"
    assert predict_category("Credit card bill paid 12000")["predicted_category"] == "EMI & Loans"
    assert predict_category("Got 1500 cashback from Amazon Pay")["predicted_category"] == "Cashback"
    assert predict_category("Spent 35k on MacBook downpayment")["predicted_category"] == "Shopping"

    cb_parse = parse_natural_language_expense("Got 1500 cashback from Amazon Pay")
    assert cb_parse["transaction_type"] == "credit"
    assert not cb_parse["description"].lower().startswith("got ")

    # 3. Debt payoff with extra_monthly=0 yields 0 interest saved
    db_session.add(Loan(
        user_id=test_user.id,
        name="Personal Loan",
        loan_type=LoanType.PERSONAL,
        principal_amount=Decimal("100000.00"),
        outstanding_amount=Decimal("100000.00"),
        interest_rate=14.0,
        emi_amount=Decimal("5000.00"),
        total_emis=24,
        emis_paid=0,
        status=LoanStatus.ACTIVE,
    ))
    await db_session.commit()

    dp_zero = await client.get("/api/pillars/defense/debt-payoff?extra_monthly=0", headers=auth_headers)
    assert dp_zero.status_code == 200
    assert dp_zero.json()["interest_saved"] == 0.0
    assert dp_zero.json()["months_saved"] == 0

    # 4. Agent routes casual purchase phrasing to simulate_purchase
    ag_buy = await client.post("/api/agent/chat", json={"message": "Can I buy shoes for 2500?"}, headers=auth_headers)
    assert ag_buy.status_code == 200
    assert "simulate_purchase" in ag_buy.json()["tools_used"]
