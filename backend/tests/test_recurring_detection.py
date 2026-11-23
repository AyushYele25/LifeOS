"""
Tests for Phase 4 — Recurring Payment Detection:
- Monthly fixed subscriptions
- Variable utility bills
- Annual renewals
- Rejection of irregular spending
- API endpoints (detection trigger, confirmation, dismissal)
"""

import pytest
from datetime import date, timedelta
from app.services.recurring_detector import RecurringDetector, normalize_merchant_name
from app.models.models import SubscriptionFrequency


def test_merchant_name_normalization():
    """Verify various banking narration formats normalize to canonical merchant."""
    assert normalize_merchant_name(None, "POS/E-COM RECURRING — NETFLIX INDIA") == "Netflix"
    assert normalize_merchant_name(None, "UPI/PAYMENT TO SPOTIFY/REF-123456") == "Spotify"
    assert normalize_merchant_name(None, "BBPS ELECTRICITY BILL — BESCOM") == "BESCOM Electricity"
    assert normalize_merchant_name(None, "UPI/HOUSE RENT/Rahul/LANDLORD") == "House Rent"


def test_monthly_fixed_subscription_detection():
    """Verify monthly fixed subscriptions are detected with high confidence."""
    detector = RecurringDetector()
    base_date = date(2026, 1, 15)

    txns = [
        {"date": base_date, "amount": 649.0, "description": "POS/E-COM RECURRING — NETFLIX INDIA", "merchant": "Netflix India"},
        {"date": base_date + timedelta(days=30), "amount": 649.0, "description": "POS/E-COM RECURRING — NETFLIX INDIA", "merchant": "Netflix India"},
        {"date": base_date + timedelta(days=61), "amount": 649.0, "description": "POS/E-COM RECURRING — NETFLIX INDIA", "merchant": "Netflix India"},
        {"date": base_date + timedelta(days=91), "amount": 649.0, "description": "POS/E-COM RECURRING — NETFLIX INDIA", "merchant": "Netflix India"},
    ]

    candidates = detector.analyze_transactions(txns)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["merchant"] == "Netflix"
    assert cand["frequency"] == SubscriptionFrequency.MONTHLY
    assert cand["confidence"] >= 0.85
    assert float(cand["average_amount"]) == 649.0


def test_variable_utility_bill_detection():
    """Verify monthly variable bills (e.g. electricity) are detected with varying amounts."""
    detector = RecurringDetector()
    base_date = date(2026, 1, 10)

    txns = [
        {"date": base_date, "amount": 1450.0, "description": "BBPS ELECTRICITY BILL — BESCOM", "merchant": "Electricity Board"},
        {"date": base_date + timedelta(days=31), "amount": 1620.0, "description": "BBPS ELECTRICITY BILL — BESCOM", "merchant": "Electricity Board"},
        {"date": base_date + timedelta(days=60), "amount": 1850.0, "description": "BBPS ELECTRICITY BILL — BESCOM", "merchant": "Electricity Board"},
        {"date": base_date + timedelta(days=90), "amount": 1500.0, "description": "BBPS ELECTRICITY BILL — BESCOM", "merchant": "Electricity Board"},
    ]

    candidates = detector.analyze_transactions(txns)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["frequency"] == SubscriptionFrequency.MONTHLY
    assert cand["confidence"] >= 0.70
    assert float(cand["amount_min"]) == 1450.0
    assert float(cand["amount_max"]) == 1850.0


def test_annual_renewal_detection():
    """Verify annual renewals (~365 days apart) are detected."""
    detector = RecurringDetector()
    txns = [
        {"date": date(2024, 6, 1), "amount": 1499.0, "description": "AMAZON PRIME ANNUAL", "merchant": "Amazon Prime"},
        {"date": date(2025, 6, 2), "amount": 1499.0, "description": "AMAZON PRIME ANNUAL", "merchant": "Amazon Prime"},
        {"date": date(2026, 6, 1), "amount": 1499.0, "description": "AMAZON PRIME ANNUAL", "merchant": "Amazon Prime"},
    ]

    candidates = detector.analyze_transactions(txns)
    assert len(candidates) == 1
    assert candidates[0]["frequency"] == SubscriptionFrequency.YEARLY


def test_irregular_spending_rejected():
    """Verify irregular dining/shopping charges are NOT marked as recurring."""
    detector = RecurringDetector()
    base_date = date(2026, 1, 1)

    # 4 Swiggy orders on irregular dates with wild amount variances
    txns = [
        {"date": base_date, "amount": 350.0, "description": "UPI/SWIGGY/1", "merchant": "Swiggy"},
        {"date": base_date + timedelta(days=3), "amount": 1200.0, "description": "UPI/SWIGGY/2", "merchant": "Swiggy"},
        {"date": base_date + timedelta(days=19), "amount": 220.0, "description": "UPI/SWIGGY/3", "merchant": "Swiggy"},
        {"date": base_date + timedelta(days=44), "amount": 800.0, "description": "UPI/SWIGGY/4", "merchant": "Swiggy"},
    ]

    candidates = detector.analyze_transactions(txns)
    # Intervals are 3, 16, 25 days -> does not match any regular weekly/monthly cadence
    assert len(candidates) == 0


@pytest.mark.asyncio
async def test_recurring_api_endpoints(client):
    """Test recurring payments API: trigger detection, confirm, and dismiss."""
    reg = await client.post("/api/auth/register", json={
        "email": "recurring_user@example.com",
        "password": "Password123!",
        "full_name": "Recurring Tester",
    })
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # Add 4 monthly Spotify transactions
    base_date = date(2026, 1, 5)
    for i in range(4):
        await client.post("/api/transactions", headers=headers, json={
            "date": str(base_date + timedelta(days=i * 30)),
            "description": "POS/E-COM RECURRING — SPOTIFY INDIA",
            "amount": 119.0,
            "transaction_type": "debit"
        })

    # 1. Trigger detection via POST /api/recurring-payments/detect
    detect_res = await client.post("/api/recurring-payments/detect", headers=headers)
    assert detect_res.status_code == 200
    data = detect_res.json()
    assert data["detected_count"] >= 1
    pattern = data["patterns"][0]
    assert pattern["merchant"] == "Spotify"
    assert pattern["is_confirmed"] is False
    pattern_id = pattern["id"]

    # 2. List patterns via GET /api/recurring-payments
    list_res = await client.get("/api/recurring-payments", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 3. Confirm pattern via PATCH /api/recurring-payments/{id}/confirm
    conf_res = await client.patch(f"/api/recurring-payments/{pattern_id}/confirm", headers=headers)
    assert conf_res.status_code == 200
    assert conf_res.json()["is_confirmed"] is True

    # 4. Dismiss pattern via PATCH /api/recurring-payments/{id}/dismiss
    dismiss_res = await client.patch(f"/api/recurring-payments/{pattern_id}/dismiss", headers=headers)
    assert dismiss_res.status_code == 204

    # Verify dismissed
    list_res_after = await client.get("/api/recurring-payments", headers=headers)
    assert len(list_res_after.json()) == 0
