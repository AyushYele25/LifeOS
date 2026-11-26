"""
LIFEOS Pillars API Routes

Exposes:
1. Daily Safe-to-Spend Metric
2. Autonomous Financial Defense (Zombie subscriptions, Debt Snowball/Avalanche, Card Rewards)
3. Indian Tax & Salary Structuring Copilot
4. Zero-Friction Voice/Text & Receipt Ingestion
5. Household Shared Vault & Split Ledger
"""

from datetime import date, timedelta
from decimal import Decimal
import uuid as _uuid
from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from app.database import get_db
from app.models import (
    User, Account, Transaction, Category, Subscription, Goal,
    HouseholdVault, VaultExpense,
)
from app.models.models import (
    AccountType, TransactionType, SubscriptionFrequency, SubscriptionStatus, GoalStatus,
)
from app.services.auth_service import get_current_user
from app.services.life_state_engine import LifeStateEngine
from app.services.event_engine import EventEngine
from app.services.safe_to_spend import compute_safe_to_spend
from app.services.financial_defense import (
    analyze_zombie_subscriptions,
    compute_debt_payoff_strategies,
    get_credit_card_recommendation,
)
from app.services.tax_engine import compare_tax_regimes
from app.services.quick_ingest import parse_natural_language_expense, parse_receipt_text

router = APIRouter(prefix="/api/pillars", tags=["Pillars & Innovation"])


# ─── QUICK 30-SECOND REAL-LIFE ONBOARDING SETUP ──────────────
class QuickSetupRequest(BaseModel):
    liquid_balance: float = Field(..., ge=0)
    monthly_income: float = Field(..., ge=0)
    fixed_monthly_bills: float = Field(0.0, ge=0)
    account_name: str = Field("Primary Bank Account", min_length=1, max_length=255)
    goal_name: Optional[str] = None
    goal_target: Optional[float] = Field(None, gt=0)


@router.post("/quick-setup")
async def run_quick_setup(
    data: QuickSetupRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Sets up a real user's financial baseline in one atomic step:
    - Creates or updates primary liquid account balance
    - Records monthly income credit (if > 0)
    - Records monthly fixed rent/bills subscription/expense (if > 0)
    - Creates an optional goal
    - Recomputes LifeState and syncs Upcoming Obligations
    """
    today = date.today()

    # 1. Account
    acc_res = await db.execute(
        select(Account).where(
            and_(Account.user_id == current_user.id, Account.is_active == True)
        ).order_by(Account.is_primary.desc(), Account.created_at.asc()).limit(1)
    )
    account = acc_res.scalar_one_or_none()
    if account:
        account.current_balance = Decimal(str(round(data.liquid_balance, 2)))
        account.name = data.account_name
    else:
        account = Account(
            user_id=current_user.id,
            name=data.account_name,
            account_type=AccountType.SAVINGS,
            bank_name="Primary Bank",
            current_balance=Decimal(str(round(data.liquid_balance, 2))),
            is_primary=True,
        )
        db.add(account)
        await db.flush()

    # 2. Monthly Income Credit
    if data.monthly_income > 0:
        sal_cat_res = await db.execute(
            select(Category).where(Category.name == "Salary").limit(1)
        )
        sal_cat = sal_cat_res.scalar_one_or_none()
        db.add(Transaction(
            user_id=current_user.id,
            account_id=account.id,
            category_id=sal_cat.id if sal_cat else None,
            date=today,
            description="Monthly Income / Allowance",
            amount=Decimal(str(round(data.monthly_income, 2))),
            transaction_type=TransactionType.CREDIT,
            merchant="Monthly Income",
            source="quick_setup",
        ))

    # 3. Fixed Monthly Bills / Rent
    if data.fixed_monthly_bills > 0:
        rent_cat_res = await db.execute(
            select(Category).where(Category.name == "Rent").limit(1)
        )
        rent_cat = rent_cat_res.scalar_one_or_none()
        db.add(Transaction(
            user_id=current_user.id,
            account_id=account.id,
            category_id=rent_cat.id if rent_cat else None,
            date=today,
            description="Monthly Fixed Rent & Bills",
            amount=Decimal(str(round(data.fixed_monthly_bills, 2))),
            transaction_type=TransactionType.DEBIT,
            merchant="Fixed Obligations",
            source="quick_setup",
        ))
        db.add(Subscription(
            user_id=current_user.id,
            name="Fixed Rent & Utilities Baseline",
            provider="Monthly Fixed Bills",
            amount=Decimal(str(round(data.fixed_monthly_bills, 2))),
            frequency=SubscriptionFrequency.MONTHLY,
            start_date=today,
            next_billing_date=today + timedelta(days=30),
            category="Rent",
            status=SubscriptionStatus.ACTIVE,
        ))

    # 4. Optional Milestone Goal
    if data.goal_name and data.goal_target and data.goal_target > 0:
        db.add(Goal(
            user_id=current_user.id,
            name=data.goal_name.strip(),
            target_amount=Decimal(str(round(data.goal_target, 2))),
            current_amount=Decimal("0.00"),
            priority=1,
            status=GoalStatus.ACTIVE,
        ))

    await db.flush()

    lse = LifeStateEngine(db, current_user.id)
    await lse.recompute_state()
    ev_engine = EventEngine(db, current_user.id)
    await ev_engine.sync_all_obligations(horizon_days=90)
    await db.commit()

    return {
        "status": "configured",
        "account_id": str(account.id),
        "safe_to_spend": await compute_safe_to_spend(current_user.id, db),
    }


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


# ─── PILLAR 6: HOUSEHOLD SHARED VAULT (DB-BACKED PER USER) ───
class AddSharedExpenseRequest(BaseModel):
    description: str
    amount: float = Field(..., gt=0)
    paid_by: str
    category: str = "Household Services"


class UpdateVaultMembersRequest(BaseModel):
    vault_name: Optional[str] = None
    members: List[str] = Field(..., min_length=2)


async def _get_or_init_user_vault(current_user: User, db: AsyncSession) -> HouseholdVault:
    res = await db.execute(
        select(HouseholdVault).where(HouseholdVault.user_id == current_user.id)
    )
    vault = res.scalar_one_or_none()
    if vault:
        return vault

    is_demo = current_user.email == "demo.priya@lifeos.ai"
    if is_demo:
        vault = HouseholdVault(
            user_id=current_user.id,
            vault_name="Koregaon Park Flat 402 Shared Vault",
            members=["Priya Sharma", "Rohan Mehta", "Ananya Verma"],
        )
        db.add(vault)
        demo_rows = [
            VaultExpense(user_id=current_user.id, description="Urban Company Maid & Cook Salary", amount=Decimal("12000.00"), paid_by="Priya Sharma", category="Household Services", expense_date=date.today() - timedelta(days=15)),
            VaultExpense(user_id=current_user.id, description="Tata Power Residential Electricity Bill", amount=Decimal("3450.00"), paid_by="Rohan Mehta", category="Electricity", expense_date=date.today() - timedelta(days=9)),
            VaultExpense(user_id=current_user.id, description="Blinkit Monthly Grocery Restock", amount=Decimal("4200.00"), paid_by="Ananya Verma", category="Groceries", expense_date=date.today() - timedelta(days=4)),
            VaultExpense(user_id=current_user.id, description="Airtel Xstream Fiber 300Mbps Wifi", amount=Decimal("1499.00"), paid_by="Priya Sharma", category="Mobile & Internet", expense_date=date.today() - timedelta(days=1)),
        ]
        for row in demo_rows:
            db.add(row)
    else:
        owner_name = (current_user.full_name or "You").strip()
        vault = HouseholdVault(
            user_id=current_user.id,
            vault_name=f"{owner_name.split()[0]}'s Shared Vault",
            members=[owner_name, "Partner / Flatmate"],
        )
        db.add(vault)

    await db.flush()
    await db.refresh(vault)
    return vault


@router.get("/household/vault")
async def get_household_vault(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetches per-user shared expenses from DB and computes who-owes-whom settlements."""
    vault = await _get_or_init_user_vault(current_user, db)
    members = list(vault.members or [])
    if len(members) < 2:
        members = [current_user.full_name or "You", "Partner / Flatmate"]

    exp_res = await db.execute(
        select(VaultExpense)
        .where(and_(VaultExpense.user_id == current_user.id, VaultExpense.is_settled == False))
        .order_by(desc(VaultExpense.expense_date), desc(VaultExpense.created_at))
    )
    rows = exp_res.scalars().all()
    expenses = [
        {
            "id": str(r.id),
            "description": r.description,
            "amount": float(r.amount),
            "paid_by": r.paid_by,
            "split_between": members,
            "category": r.category,
            "date": str(r.expense_date),
        }
        for r in rows
    ]

    total_spent = sum(e["amount"] for e in expenses)
    per_person_share = total_spent / len(members) if members else 0.0

    paid_map = {m: 0.0 for m in members}
    for e in expenses:
        paid_map[e["paid_by"]] = paid_map.get(e["paid_by"], 0.0) + e["amount"]

    balances = {m: round(paid_map.get(m, 0.0) - per_person_share, 2) for m in members}

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
        "vault_name": vault.vault_name,
        "members": members,
        "total_shared_spend": round(total_spent, 2),
        "per_person_share": round(per_person_share, 2),
        "balances": balances,
        "settlements": settlements,
        "expenses": expenses,
    }


@router.put("/household/members")
async def update_household_members(
    data: UpdateVaultMembersRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Updates group name and member names for the authenticated user's shared vault."""
    vault = await _get_or_init_user_vault(current_user, db)
    clean_members = [m.strip() for m in data.members if m.strip()]
    if len(clean_members) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 member names")
    vault.members = clean_members
    if data.vault_name and data.vault_name.strip():
        vault.vault_name = data.vault_name.strip()
    await db.flush()
    return await get_household_vault(current_user, db)


@router.post("/household/split")
async def add_shared_expense(
    data: AddSharedExpenseRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Adds a shared expense to the user's persistent vault and recomputes balances."""
    vault = await _get_or_init_user_vault(current_user, db)
    members = list(vault.members or [])
    if data.paid_by.strip() and data.paid_by.strip() not in members:
        members.append(data.paid_by.strip())
        vault.members = members

    db.add(VaultExpense(
        user_id=current_user.id,
        description=data.description.strip(),
        amount=Decimal(str(round(data.amount, 2))),
        paid_by=data.paid_by.strip(),
        category=data.category,
        expense_date=date.today(),
    ))
    await db.flush()
    return await get_household_vault(current_user, db)


@router.delete("/household/expense/{expense_id}")
async def delete_shared_expense(
    expense_id: _uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Removes a shared expense from the authenticated user's vault."""
    res = await db.execute(
        select(VaultExpense).where(
            and_(VaultExpense.id == expense_id, VaultExpense.user_id == current_user.id)
        )
    )
    exp = res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Shared expense not found")
    await db.delete(exp)
    await db.flush()
    return await get_household_vault(current_user, db)
