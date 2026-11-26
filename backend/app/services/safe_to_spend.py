"""
LIFEOS Safe-to-Spend Engine

Calculates dynamic daily discretionary allowance based on monthly income,
contractual fixed obligations (rent, loans, subscriptions), and wealth goal targets.
"""

from datetime import date
import calendar
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc

from app.models import LifeState, Transaction, Goal, Loan, Subscription, Account
from app.models.models import TransactionType, LoanStatus, SubscriptionStatus, GoalStatus

FIXED_CATEGORIES = {
    "Rent",
    "EMI & Loans",
    "Utilities",
    "Electricity",
    "Mobile & Internet",
    "Insurance",
    "Subscriptions",
    "Groceries",
}


async def compute_safe_to_spend(
    user_id: Any, db: AsyncSession, target_date: date = None
) -> Dict[str, Any]:
    """
    Computes real-time daily safe-to-spend allowance:
    Formula:
        Monthly Income (or liquid reserve allowance if income not logged yet)
      - Contractual Fixed Floor (Rent + Loans + Subscriptions + Utilities)
      - Planned Wealth & Goal Savings Target
      = Free Discretionary Pool

      Dynamic Daily Allowance = (Remaining Discretionary Pool) / (Remaining Days in Month)
    """
    today = target_date or date.today()
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    day_of_month = today.day
    days_remaining = max(1, days_in_month - day_of_month + 1)
    month_start = today.replace(day=1)

    # 1. Fetch latest life state and live account balances
    ls_result = await db.execute(
        select(LifeState)
        .where(LifeState.user_id == user_id)
        .order_by(desc(LifeState.snapshot_date))
        .limit(1)
    )
    life_state = ls_result.scalar_one_or_none()

    acc_result = await db.execute(
        select(func.coalesce(func.sum(Account.current_balance), 0)).where(
            and_(Account.user_id == user_id, Account.is_active == True)
        )
    )
    live_balance = float(acc_result.scalar() or 0.0)

    mtd_credit_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.CREDIT,
                Transaction.date >= month_start,
                Transaction.date <= today,
            )
        )
    )
    mtd_credits = float(mtd_credit_result.scalar() or 0.0)

    # 2. Contractual Fixed Floor & MTD spend
    loans_result = await db.execute(
        select(func.coalesce(func.sum(Loan.emi_amount), 0)).where(
            and_(Loan.user_id == user_id, Loan.status == LoanStatus.ACTIVE)
        )
    )
    loans_monthly = float(loans_result.scalar() or 0.0)

    subs_result = await db.execute(
        select(func.coalesce(func.sum(Subscription.amount), 0)).where(
            and_(
                Subscription.user_id == user_id,
                Subscription.status == SubscriptionStatus.ACTIVE,
            )
        )
    )
    subs_monthly = float(subs_result.scalar() or 0.0)

    mtd_spend_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.DEBIT,
                Transaction.date >= month_start,
                Transaction.date <= today,
            )
        )
    )
    mtd_total_spend = float(mtd_spend_result.scalar() or 0.0)

    ls_income = (
        float(life_state.monthly_income)
        if life_state and life_state.monthly_income
        else 0.0
    )
    recorded_income = max(ls_income, mtd_credits)
    ls_balance = (
        float(life_state.total_balance)
        if life_state and life_state.total_balance
        else 0.0
    )
    total_balance = max(live_balance, ls_balance)

    # If user has completely empty data, return honest setup prompt instead of fake defaults
    if (
        recorded_income <= 0
        and total_balance <= 0
        and loans_monthly <= 0
        and subs_monthly <= 0
        and mtd_total_spend <= 0
    ):
        return {
            "today_date": str(today),
            "days_in_month": days_in_month,
            "day_of_month": day_of_month,
            "days_remaining": days_remaining,
            "monthly_income": 0.0,
            "fixed_obligations_total": 0.0,
            "monthly_goal_savings": 0.0,
            "discretionary_pool_total": 0.0,
            "baseline_daily_allowance": 0.0,
            "mtd_discretionary_spend": 0.0,
            "remaining_discretionary_pool": 0.0,
            "daily_safe_to_spend": 0.0,
            "today_spent": 0.0,
            "today_remaining_allowance": 0.0,
            "needs_setup": True,
            "status_code": "NEEDS_SETUP",
            "status_label": "⚙️ Setup Needed",
            "status_color": "amber",
            "advice": "Enter your bank balance and monthly income in the 30-second Quick Setup below to unlock your real daily spending limit.",
        }

    income_estimated = False
    if recorded_income > 0:
        monthly_income = recorded_income
    elif total_balance > 0:
        # Estimate a sustainable monthly spending envelope (1/3 of liquid reserve) when salary isn't logged yet
        monthly_income = total_balance / 3.0
        income_estimated = True
    else:
        monthly_income = 0.0

    # Estimate fixed living baseline from LifeState breakdown or contractual obligations
    fixed_from_categories = 0.0
    if life_state and life_state.expense_by_category:
        cat_dict = life_state.expense_by_category
        total_cat_spend = sum(float(v) for v in cat_dict.values())
        fixed_cat_spend = sum(
            float(v) for k, v in cat_dict.items() if k in FIXED_CATEGORIES
        )
        if total_cat_spend > 0 and life_state.monthly_expenses:
            fixed_ratio = fixed_cat_spend / total_cat_spend
            fixed_from_categories = float(life_state.monthly_expenses) * fixed_ratio

    baseline_fixed = max(
        fixed_from_categories,
        loans_monthly + subs_monthly,
        float(life_state.monthly_expenses) * 0.7 if life_state and life_state.monthly_expenses else 0.0,
    )
    fixed_obligations_total = min(monthly_income * 0.85, baseline_fixed) if monthly_income > 0 else baseline_fixed

    # 3. Monthly Goal Targets
    goals_result = await db.execute(
        select(Goal).where(
            and_(Goal.user_id == user_id, Goal.status == GoalStatus.ACTIVE)
        )
    )
    goals = goals_result.scalars().all()
    monthly_goal_savings = (
        monthly_income * 0.20
        if not goals
        else sum(
            float(g.target_amount - g.current_amount) / 12.0
            for g in goals
            if g.target_amount > g.current_amount
        )
    )
    monthly_goal_savings = min(
        monthly_goal_savings, max(0.0, (monthly_income - fixed_obligations_total) * 0.5)
    )

    # 4. Total Discretionary Pool for Month
    discretionary_pool_total = max(
        0.0, monthly_income - fixed_obligations_total - monthly_goal_savings
    )
    baseline_daily_allowance = discretionary_pool_total / days_in_month

    # Pro-rate fixed obligations over elapsed days to isolate discretionary burn
    pro_rated_fixed = (fixed_obligations_total / days_in_month) * day_of_month
    mtd_discretionary_spend = max(0.0, mtd_total_spend - pro_rated_fixed)

    # 5. Remaining Discretionary Pool & Today's Allowance
    remaining_discretionary_pool = max(
        0.0, discretionary_pool_total - mtd_discretionary_spend
    )
    daily_safe_to_spend = remaining_discretionary_pool / days_remaining

    # 6. Today's actual spending so far
    today_spend_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.DEBIT,
                Transaction.date == today,
            )
        )
    )
    today_spent = float(today_spend_result.scalar() or 0.0)
    today_remaining_allowance = max(0.0, daily_safe_to_spend - today_spent)

    # Status determination
    if daily_safe_to_spend <= 0 and monthly_income <= 0:
        status_code = "TIGHT"
        status_label = "🔴 Income Not Logged"
        color = "rose"
    elif daily_safe_to_spend >= baseline_daily_allowance * 0.9:
        status_code = "HEALTHY"
        status_label = "🟢 On Track"
        color = "emerald"
    elif daily_safe_to_spend >= baseline_daily_allowance * 0.5:
        status_code = "MODERATE"
        status_label = "🟡 Moderate Buffer"
        color = "amber"
    else:
        status_code = "TIGHT"
        status_label = "🔴 Budget Constrained"
        color = "rose"

    if income_estimated:
        advice = (
            f"Estimated ₹{today_remaining_allowance:,.0f}/day from your liquid balance. "
            "Log your monthly salary or allowance for exact budgeting."
        )
    elif today_remaining_allowance > 0:
        advice = (
            f"You can comfortably spend up to ₹{today_remaining_allowance:,.0f} today. "
            "Your rent, EMIs, and monthly goals are fully protected."
        )
    else:
        advice = (
            f"Daily budget reached for today (₹{today_spent:,.0f} spent). "
            "Limit non-essential expenses until tomorrow morning."
        )

    return {
        "today_date": str(today),
        "days_in_month": days_in_month,
        "day_of_month": day_of_month,
        "days_remaining": days_remaining,
        "monthly_income": round(monthly_income, 2),
        "fixed_obligations_total": round(fixed_obligations_total, 2),
        "monthly_goal_savings": round(monthly_goal_savings, 2),
        "discretionary_pool_total": round(discretionary_pool_total, 2),
        "baseline_daily_allowance": round(baseline_daily_allowance, 2),
        "mtd_discretionary_spend": round(mtd_discretionary_spend, 2),
        "remaining_discretionary_pool": round(remaining_discretionary_pool, 2),
        "daily_safe_to_spend": round(daily_safe_to_spend, 2),
        "today_spent": round(today_spent, 2),
        "today_remaining_allowance": round(today_remaining_allowance, 2),
        "needs_setup": False,
        "income_estimated": income_estimated,
        "status_code": status_code,
        "status_label": status_label,
        "status_color": color,
        "advice": advice,
    }
