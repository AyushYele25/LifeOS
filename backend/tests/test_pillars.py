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
from datetime import date

from app.services.quick_ingest import parse_natural_language_expense, parse_receipt_text
from app.services.tax_engine import compare_tax_regimes, calculate_new_regime_tax, calculate_old_regime_tax
from app.services.financial_defense import get_credit_card_recommendation


def test_quick_natural_language_expense_parsing():
    """Pillar 1: Test parsing conversational voice/text strings."""
    # Test food expense
    res1 = parse_natural_language_expense("Spent 450 at CCD on coffee via HDFC")
    assert res1["amount"] == 450.0
    assert res1["transaction_type"] == "debit"
    assert "Food" in res1["predicted_category"] or "Dining" in res1["predicted_category"]

    # Test fuel expense
    res2 = parse_natural_language_expense("Petrol 2200 at BPCL pump")
    assert res2["amount"] == 2200.0
    assert res2["transaction_type"] == "debit"
    assert res2["predicted_category"] == "Fuel"

    # Test salary credit
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
    assert "Starbucks" in res["merchant"]
    assert "Food" in res["predicted_category"] or "Dining" in res["predicted_category"]


def test_tax_engine_regime_comparison():
    """Pillar 5: Test Indian income tax calculations for Old vs New Regime."""
    # Person earning ₹15 Lakhs with standard ₹1.5L 80C, ₹50k NPS, ₹1.8L HRA
    res = compare_tax_regimes(
        gross_income=1500000.0,
        hra_exemption=180000.0,
        sec_80c=150000.0,
        sec_80d=25000.0,
        sec_80ccd_nps=50000.0
    )
    assert res["gross_income"] == 1500000.0
    assert res["new_regime"]["total_tax_payable"] > 0
    assert res["old_regime"]["total_tax_payable"] > 0
    assert "winner" in res
    assert res["annual_tax_savings"] >= 0


def test_credit_card_rewards_optimizer():
    """Pillar 4: Test credit card recommendation for merchant categories."""
    swiggy_res = get_credit_card_recommendation("Food & Dining", 850.0)
    assert "HDFC" in swiggy_res["recommended_card"] or "Cashback" in swiggy_res["recommended_card"]
    assert swiggy_res["estimated_cashback"] > 0

    amazon_res = get_credit_card_recommendation("Shopping", 5000.0)
    assert "ICICI Amazon" in amazon_res["recommended_card"] or "SBI Cashback" in amazon_res["recommended_card"]
    assert amazon_res["estimated_cashback"] > 0


@pytest.mark.asyncio
async def test_pillars_api_endpoints(client: AsyncClient, auth_headers: dict):
    """Integration test for API routes under /api/pillars."""
    # 1. Safe to Spend
    s2s_res = await client.get("/api/pillars/safe-to-spend", headers=auth_headers)
    assert s2s_res.status_code == 200
    s2s_data = s2s_res.json()
    assert "daily_safe_to_spend" in s2s_data
    assert "discretionary_pool_total" in s2s_data
    assert s2s_data["daily_safe_to_spend"] >= 0

    # 2. Defense - Zombies
    z_res = await client.get("/api/pillars/defense/zombies", headers=auth_headers)
    assert z_res.status_code == 200
    z_data = z_res.json()
    assert "total_active_subscriptions" in z_data

    # 3. Defense - Debt Payoff
    d_res = await client.get("/api/pillars/defense/debt-payoff?extra_monthly=3000", headers=auth_headers)
    assert d_res.status_code == 200
    d_data = d_res.json()
    assert "has_debt" in d_data

    # 4. Tax Compare Endpoint
    tax_payload = {
        "gross_annual_income": 1200000.0,
        "hra_exemption": 120000.0,
        "sec_80c": 150000.0,
        "sec_80d": 25000.0,
        "sec_80ccd_nps": 50000.0
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

