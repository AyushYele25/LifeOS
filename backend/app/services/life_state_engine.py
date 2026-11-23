"""
LIFEOS Life State Engine

The core deterministic financial calculation engine of LIFEOS:
- Liquid balance, net worth, and asset valuation
- Monthly income, expenses, and savings rate
- Contractual and predictable obligations (30/60/90 days)
- Emergency runway (months of living expenses covered)
- Debt-to-Income (DTI) and composite Financial Risk Score (0-100)
- Historical snapshotting and state delta comparisons ("What changed?")
"""

from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import uuid

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.models import (
    User,
    Account,
    AccountType,
    Transaction,
    TransactionType,
    Loan,
    LoanStatus,
    Insurance,
    Subscription,
    SubscriptionStatus,
    Asset,
    Event,
    LifeState,
    RiskLevel,
    AuditLog,
)
from app.services.event_engine import EventEngine


class LifeStateEngine:
    """
    Deterministic engine that computes the user's comprehensive financial health.
    """

    def __init__(self, db: AsyncSession, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id

    async def recompute_state(self, save_snapshot: bool = True) -> LifeState:
        """
        Executes deterministic calculations and produces a complete LifeState snapshot.
        """
        # 1. Money — Total Liquid Balance
        acc_query = select(Account).where(
            and_(Account.user_id == self.user_id, Account.is_active == True)
        )
        acc_res = await self.db.execute(acc_query)
        accounts = acc_res.scalars().all()

        total_balance = Decimal("0.00")
        if accounts:
            for acc in accounts:
                bal = acc.current_balance or Decimal("0.00")
                if acc.account_type == AccountType.CREDIT_CARD:
                    total_balance -= bal  # Credit card debt reduces liquid net
                else:
                    total_balance += bal
        # 2. Fetch Transactions for Income & Expenses
        tx_query = (
            select(Transaction)
            .options(selectinload(Transaction.category))
            .where(Transaction.user_id == self.user_id)
            .order_by(Transaction.date.asc())
        )
        tx_res = await self.db.execute(tx_query)
        transactions = tx_res.scalars().all()

        if not accounts and transactions:
            # Fallback balance from net transactions
            net = sum(
                (t.amount if t.transaction_type == TransactionType.CREDIT else -t.amount)
                for t in transactions if t.amount
            )
            total_balance = max(Decimal("0.00"), net)

        monthly_income = Decimal("0.00")
        monthly_expenses = Decimal("0.00")
        expense_by_category: Dict[str, float] = {}
        income_by_source: Dict[str, float] = {}

        if transactions:
            monthly_credits: Dict[Tuple[int, int], Decimal] = {}
            monthly_debits: Dict[Tuple[int, int], Decimal] = {}

            for t in transactions:
                amt = t.amount or Decimal("0.00")
                cat_name = t.category.name if t.category else "Uncategorized"
                m_key = (t.date.year, t.date.month)

                if t.transaction_type == TransactionType.CREDIT:
                    monthly_credits[m_key] = monthly_credits.get(m_key, Decimal("0.00")) + amt
                    income_by_source[cat_name] = income_by_source.get(cat_name, 0.0) + float(amt)
                else:
                    monthly_debits[m_key] = monthly_debits.get(m_key, Decimal("0.00")) + amt
                    expense_by_category[cat_name] = expense_by_category.get(cat_name, 0.0) + float(amt)

            credit_months = len(monthly_credits) or 1
            debit_months = len(monthly_debits) or 1

            monthly_income = (sum(monthly_credits.values()) / Decimal(str(credit_months))).quantize(Decimal("0.01"))
            monthly_expenses = (sum(monthly_debits.values()) / Decimal(str(debit_months))).quantize(Decimal("0.01"))

        # Monthly savings & savings rate
        monthly_savings = monthly_income - monthly_expenses
        savings_rate = (
            float(monthly_savings / monthly_income)
            if monthly_income > Decimal("0.00")
            else 0.0
        )

        # 3. Obligations — Event Horizon (30/60/90 days)
        event_engine = EventEngine(self.db, self.user_id)
        horizon = await event_engine.get_horizon_summary(horizon_days=90)
        upcoming_30d = horizon["totals"]["total_30d"]
        upcoming_60d = horizon["totals"]["total_60d"]
        upcoming_90d = horizon["totals"]["total_90d"]

        # 4. Debt & Loans
        loan_res = await self.db.execute(
            select(Loan).where(
                and_(Loan.user_id == self.user_id, Loan.status == LoanStatus.ACTIVE)
            )
        )
        active_loans = loan_res.scalars().all()
        active_loans_count = len(active_loans)
        total_emi_burden = sum(
            (l.emi_amount or Decimal("0.00") for l in active_loans), Decimal("0.00")
        )

        # 5. Subscriptions
        sub_res = await self.db.execute(
            select(Subscription).where(
                and_(
                    Subscription.user_id == self.user_id,
                    Subscription.status == SubscriptionStatus.ACTIVE,
                )
            )
        )
        active_subs = sub_res.scalars().all()
        active_subscriptions_count = len(active_subs)
        total_sub_burden = sum(
            (s.amount or Decimal("0.00") for s in active_subs), Decimal("0.00")
        )

        # 6. Assets
        asset_res = await self.db.execute(
            select(Asset).where(
                and_(Asset.user_id == self.user_id, Asset.is_active == True)
            )
        )
        active_assets = asset_res.scalars().all()
        total_assets_count = len(active_assets)
        total_asset_value = sum(
            (
                a.current_value or a.purchase_price or Decimal("0.00")
                for a in active_assets
            ),
            Decimal("0.00"),
        )

        # 7. Emergency Runway (months)
        if monthly_expenses > Decimal("0.00"):
            emergency_runway_months = round(float(total_balance / monthly_expenses), 2)
        else:
            emergency_runway_months = 99.0 if total_balance > Decimal("0.00") else 0.0

        # 8. Financial Risk Score (0-100) & Breakdown
        risk_score, risk_lvl, risk_breakdown = self._compute_risk_score(
            runway=emergency_runway_months,
            income=monthly_income,
            expenses=monthly_expenses,
            emi_burden=total_emi_burden,
            balance=total_balance,
            obligations_30d=upcoming_30d,
        )

        # 9. Changes from Previous State Snapshot
        prev_state = await self.get_latest_snapshot()
        changes = self._compute_state_deltas(prev_state, {
            "total_balance": total_balance,
            "monthly_expenses": monthly_expenses,
            "monthly_income": monthly_income,
            "emergency_runway_months": emergency_runway_months,
            "financial_risk_score": risk_score,
            "upcoming_obligations_30d": upcoming_30d,
        })

        # 10. Instantiate LifeState
        state = LifeState(
            user_id=self.user_id,
            snapshot_date=datetime.now(timezone.utc),
            total_balance=total_balance,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            monthly_savings=monthly_savings,
            savings_rate=round(savings_rate, 4),
            upcoming_obligations_30d=upcoming_30d,
            upcoming_obligations_60d=upcoming_60d,
            upcoming_obligations_90d=upcoming_90d,
            active_loans_count=active_loans_count,
            total_emi_burden=total_emi_burden,
            active_subscriptions_count=active_subscriptions_count,
            total_subscription_burden=total_sub_burden,
            total_assets_count=total_assets_count,
            total_asset_value=total_asset_value,
            financial_risk_score=risk_score,
            risk_level=risk_lvl,
            emergency_runway_months=emergency_runway_months,
            risk_breakdown=risk_breakdown,
            expense_by_category=expense_by_category,
            income_by_source=income_by_source,
            changes_from_previous=changes,
        )

        if save_snapshot:
            self.db.add(state)
            audit = AuditLog(
                user_id=self.user_id,
                action="life_state_snapshot",
                entity_type="life_state",
                entity_id=str(state.id),
                details={
                    "balance": float(total_balance),
                    "runway": emergency_runway_months,
                    "risk_score": risk_score,
                },
            )
            self.db.add(audit)
            await self.db.commit()
            await self.db.refresh(state)

        return state

    def _compute_risk_score(
        self,
        runway: float,
        income: Decimal,
        expenses: Decimal,
        emi_burden: Decimal,
        balance: Decimal,
        obligations_30d: Decimal,
    ) -> Tuple[float, RiskLevel, Dict[str, Any]]:
        """
        Deterministic multi-factor risk assessment (0-100 index).
        """
        score = 25.0
        breakdown = {}

        # Factor 1: Emergency Runway
        if runway < 1.0:
            score += 35.0
            breakdown["runway"] = {"points": 35.0, "reason": "Critically low emergency runway (< 1 month)"}
        elif runway < 3.0:
            score += 20.0
            breakdown["runway"] = {"points": 20.0, "reason": "Thin emergency cushion (1-3 months)"}
        elif runway < 6.0:
            score += 5.0
            breakdown["runway"] = {"points": 5.0, "reason": "Moderate liquidity buffer (3-6 months)"}
        else:
            score -= 15.0
            breakdown["runway"] = {"points": -15.0, "reason": "Robust emergency cushion (>= 6 months)"}

        # Factor 2: Debt-to-Income (DTI)
        dti = float(emi_burden / income) if income > Decimal("0.00") else 0.0
        if dti > 0.50:
            score += 30.0
            breakdown["debt_burden"] = {"points": 30.0, "reason": f"High debt burden (DTI {dti*100:.1f}%)"}
        elif dti > 0.35:
            score += 18.0
            breakdown["debt_burden"] = {"points": 18.0, "reason": f"Elevated debt-to-income (DTI {dti*100:.1f}%)"}
        elif dti > 0.20:
            score += 8.0
            breakdown["debt_burden"] = {"points": 8.0, "reason": f"Moderate debt-to-income (DTI {dti*100:.1f}%)"}
        else:
            score -= 5.0
            breakdown["debt_burden"] = {"points": -5.0, "reason": "Low or zero debt burden"}

        # Factor 3: Cashflow / Savings Rate
        if income > Decimal("0.00"):
            s_rate = float((income - expenses) / income)
            if s_rate < 0.0:
                score += 25.0
                breakdown["savings_rate"] = {"points": 25.0, "reason": "Negative cashflow: spending exceeds earnings"}
            elif s_rate < 0.10:
                score += 10.0
                breakdown["savings_rate"] = {"points": 10.0, "reason": "Low savings rate (< 10%)"}
            elif s_rate >= 0.25:
                score -= 15.0
                breakdown["savings_rate"] = {"points": -15.0, "reason": "Excellent savings rate (>= 25%)"}
            else:
                breakdown["savings_rate"] = {"points": 0.0, "reason": "Balanced savings rate"}

        # Factor 4: Immediate 30-day obligations coverage
        if obligations_30d > balance and obligations_30d > Decimal("0.00"):
            score += 25.0
            breakdown["immediate_obligations"] = {
                "points": 25.0,
                "reason": "Upcoming 30-day obligations exceed current liquid balance",
            }
        else:
            breakdown["immediate_obligations"] = {
                "points": 0.0,
                "reason": "Current liquid balance covers upcoming 30-day obligations",
            }

        # Normalize score
        final_score = round(max(0.0, min(100.0, score)), 1)

        if final_score >= 75.0:
            lvl = RiskLevel.CRITICAL
        elif final_score >= 50.0:
            lvl = RiskLevel.HIGH
        elif final_score >= 25.0:
            lvl = RiskLevel.MEDIUM
        else:
            lvl = RiskLevel.LOW

        return final_score, lvl, breakdown

    def _compute_state_deltas(
        self, prev: Optional[LifeState], current: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Computes deltas against the prior recorded state."""
        if not prev:
            return []

        deltas = []
        # Balance delta
        old_bal = float(prev.total_balance or 0)
        new_bal = float(current["total_balance"])
        diff_bal = new_bal - old_bal
        pct_bal = round((diff_bal / old_bal) * 100, 1) if old_bal != 0 else 0.0
        if abs(diff_bal) > 100:
            deltas.append({
                "field": "total_balance",
                "old": old_bal,
                "new": new_bal,
                "change": round(diff_bal, 2),
                "change_pct": pct_bal,
            })

        # Expenses delta
        old_exp = float(prev.monthly_expenses or 0)
        new_exp = float(current["monthly_expenses"])
        diff_exp = new_exp - old_exp
        pct_exp = round((diff_exp / old_exp) * 100, 1) if old_exp != 0 else 0.0
        if abs(diff_exp) > 50:
            deltas.append({
                "field": "monthly_expenses",
                "old": old_exp,
                "new": new_exp,
                "change": round(diff_exp, 2),
                "change_pct": pct_exp,
            })

        # Runway delta
        old_run = float(prev.emergency_runway_months or 0)
        new_run = float(current["emergency_runway_months"])
        if abs(new_run - old_run) >= 0.1:
            deltas.append({
                "field": "emergency_runway_months",
                "old": old_run,
                "new": new_run,
                "change": round(new_run - old_run, 2),
            })

        # Risk score delta
        old_risk = float(prev.financial_risk_score or 0)
        new_risk = float(current["financial_risk_score"])
        if abs(new_risk - old_risk) >= 1.0:
            deltas.append({
                "field": "financial_risk_score",
                "old": old_risk,
                "new": new_risk,
                "change": round(new_risk - old_risk, 1),
            })

        return deltas

    async def get_latest_snapshot(self) -> Optional[LifeState]:
        """Fetches the most recent snapshot."""
        res = await self.db.execute(
            select(LifeState)
            .where(LifeState.user_id == self.user_id)
            .order_by(desc(LifeState.snapshot_date))
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def get_current_state(self) -> LifeState:
        """
        Returns the latest snapshot, or computes a fresh one if none exists.
        """
        latest = await self.get_latest_snapshot()
        if not latest:
            return await self.recompute_state(save_snapshot=True)
        return latest

    async def get_history(self, limit: int = 12) -> List[LifeState]:
        """Returns chronological snapshots over time."""
        res = await self.db.execute(
            select(LifeState)
            .where(LifeState.user_id == self.user_id)
            .order_by(desc(LifeState.snapshot_date))
            .limit(limit)
        )
        return res.scalars().all()
