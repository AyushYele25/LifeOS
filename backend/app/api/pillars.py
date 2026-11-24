"""
LIFEOS Pillars API Routes

Exposes:
1. Daily Safe-to-Spend Metric
2. Autonomous Financial Defense (Zombie subscriptions, Debt Snowball/Avalanche, Card Rewards)
3. Indian Tax & Salary Structuring Copilot
4. Zero-Friction Voice/Text & Receipt Ingestion
5. Household Shared Vault & Split Ledger
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.database import get_db
from app.models import User
from app.services.auth_service import get_current_user
from app.services.safe_to_spend import compute_safe_to_spend
from app.services.financial_defense import (
    analyze_zombie_subscriptions,
    compute_debt_payoff_strategies,
    get_credit_card_recommendation,
)
from app.services.tax_engine import compare_tax_regimes
from app.services.quick_ingest import parse_natural_language_expense, parse_receipt_text

router = APIRouter(prefix="/api/pillars", tags=["Pillars & Innovation"])


# ─── PILLAR 1: ZERO-EFFORT INGESTION ─────────────────────────
class QuickIngestRequest(BaseModel):
    text: str

class ReceiptParseRequest(BaseModel):
    ocr_text: str


@router.post("/ingest/parse")
async def parse_conversational_expense(data: QuickIngestRequest):
    """Parses natural voice transcription or chat string into structured transaction."""
    return parse_natural_language_expense(data.text)


@router.post("/ingest/receipt")
async def parse_receipt(data: ReceiptParseRequest):
    """Parses OCR text of supermarket/restaurant bill."""
    return parse_receipt_text(data.ocr_text)


# ─── PILLAR 2: DAILY SAFE-TO-SPEND METRIC ────────────────────
@router.get("/safe-to-spend")
async def get_daily_safe_to_spend(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Calculates live daily safe-to-spend allowance with rollover."""
    return await compute_safe_to_spend(current_user.id, db)


# ─── PILLAR 4: FINANCIAL DEFENSE (ZOMBIES & DEBT PAYOFF) ─────
@router.get("/defense/zombies")
async def get_zombie_subscriptions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Detects recurring subscriptions that bleed cash and gives cancellation guides."""
    return await analyze_zombie_subscriptions(current_user.id, db)


@router.get("/defense/debt-payoff")
async def get_debt_payoff_strategies(
    extra_monthly: float = Query(2500.0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compares Debt Snowball vs Avalanche with accelerated prepayment."""
    return await compute_debt_payoff_strategies(current_user.id, db, extra_monthly=extra_monthly)


@router.get("/defense/card-rewards")
async def get_card_reward_advice(
    category: str = Query("Food & Dining"),
    amount: float = Query(1000.0, gt=0),
):
    """Recommends which Indian credit card maximizes cashback."""
    return get_credit_card_recommendation(category, amount)


# ─── PILLAR 5: INDIAN TAX COPILOT ────────────────────────────
class TaxComparisonRequest(BaseModel):
    gross_annual_income: float
    hra_exemption: float = 0.0
    sec_80c: float = 150000.0
    sec_80d: float = 25000.0
    home_loan_sec24: float = 0.0
    sec_80ccd_nps: float = 50000.0


@router.post("/tax/compare")
async def compare_tax(data: TaxComparisonRequest):
    """Compares Old vs New Regime tax liability and recommends best regime."""
    return compare_tax_regimes(
        gross_income=data.gross_annual_income,
        hra_exemption=data.hra_exemption,
        sec_80c=data.sec_80c,
        sec_80d=data.sec_80d,
        home_loan_sec24=data.home_loan_sec24,
        sec_80ccd_nps=data.sec_80ccd_nps,
    )


# ─── PILLAR 6: HOUSEHOLD SHARED VAULT ────────────────────────
# In-memory demo vault state for immediate shared split demonstration
DEMO_VAULT = {
    "vault_name": "Koregaon Park Flat 402 Shared Vault",
    "members": ["Priya Sharma", "Rohan Mehta", "Ananya Verma"],
    "expenses": [
        {"id": 1, "description": "Urban Company Maid & Cook Salary", "amount": 12000.0, "paid_by": "Priya Sharma", "split_between": ["Priya Sharma", "Rohan Mehta", "Ananya Verma"], "category": "Household Services", "date": "2026-11-01"},
        {"id": 2, "description": "Tata Power Residential Electricity Bill", "amount": 3450.0, "paid_by": "Rohan Mehta", "split_between": ["Priya Sharma", "Rohan Mehta", "Ananya Verma"], "category": "Electricity", "date": "2026-11-08"},
        {"id": 3, "description": "Blinkit Monthly Grocery Restock", "amount": 4200.0, "paid_by": "Ananya Verma", "split_between": ["Priya Sharma", "Rohan Mehta", "Ananya Verma"], "category": "Groceries", "date": "2026-11-14"},
        {"id": 4, "description": "Airtel Xstream Fiber 300Mbps Wifi", "amount": 1499.0, "paid_by": "Priya Sharma", "split_between": ["Priya Sharma", "Rohan Mehta", "Ananya Verma"], "category": "Mobile & Internet", "date": "2026-11-18"}
    ]
}

class AddSharedExpenseRequest(BaseModel):
    description: str
    amount: float
    paid_by: str
    category: str = "Household Services"


@router.get("/household/vault")
async def get_household_vault(current_user: User = Depends(get_current_user)):
    """Fetches shared household expenses and computes who-owes-whom reconciliation."""
    expenses = DEMO_VAULT["expenses"]
    members = DEMO_VAULT["members"]
    total_spent = sum(e["amount"] for e in expenses)
    per_person_share = total_spent / len(members)

    paid_map = {m: 0.0 for m in members}
    for e in expenses:
        paid_map[e["paid_by"]] = paid_map.get(e["paid_by"], 0.0) + e["amount"]

    balances = {}
    for m in members:
        balances[m] = round(paid_map[m] - per_person_share, 2)

    # Compute settlement transactions
    settlements = []
    debtors = sorted([(m, b) for m, b in balances.items() if b < 0], key=lambda x: x[1])
    creditors = sorted([(m, b) for m, b in balances.items() if b > 0], key=lambda x: x[1], reverse=True)

    d_idx = 0
    c_idx = 0
    d_list = [list(x) for x in debtors]
    c_list = [list(x) for x in creditors]

    while d_idx < len(d_list) and c_idx < len(c_list):
        debtor_name, debt_amt = d_list[d_idx]
        creditor_name, cred_amt = c_list[c_idx]
        debt_amt = abs(debt_amt)

        settle_amt = min(debt_amt, cred_amt)
        if settle_amt > 1.0:
            settlements.append({
                "from_user": debtor_name,
                "to_user": creditor_name,
                "amount": round(settle_amt, 2),
                "summary": f"{debtor_name} pays ₹{round(settle_amt):,} to {creditor_name} via UPI"
            })

        d_list[d_idx][1] += settle_amt
        c_list[c_idx][1] -= settle_amt

        if abs(d_list[d_idx][1]) < 1.0:
            d_idx += 1
        if abs(c_list[c_idx][1]) < 1.0:
            c_idx += 1

    return {
        "vault_name": DEMO_VAULT["vault_name"],
        "members": members,
        "total_shared_spend": round(total_spent, 2),
        "per_person_share": round(per_person_share, 2),
        "balances": balances,
        "settlements": settlements,
        "expenses": list(reversed(expenses)),
    }


@router.post("/household/split")
async def add_shared_expense(
    data: AddSharedExpenseRequest,
    current_user: User = Depends(get_current_user),
):
    """Adds a shared expense to the household vault and recomputes balances."""
    from datetime import date
    new_id = len(DEMO_VAULT["expenses"]) + 1
    new_entry = {
        "id": new_id,
        "description": data.description,
        "amount": data.amount,
        "paid_by": data.paid_by,
        "split_between": DEMO_VAULT["members"],
        "category": data.category,
        "date": str(date.today()),
    }
    DEMO_VAULT["expenses"].append(new_entry)
    return await get_household_vault(current_user)
