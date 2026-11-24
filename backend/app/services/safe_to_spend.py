"""
LIFEOS Safe-to-Spend Engine

Calculates dynamic daily discretionary allowance based on monthly income,
contractual fixed obligations (rent, loans, subscriptions), and wealth goal targets.
"""

from datetime import date
from decimal import Decimal
import calendar
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc

from app.models import User, LifeState, Transaction, Event, Goal, Loan, Subscription
from app.models.models import TransactionType


async def compute_safe_to_spend(user_id: Any, db: AsyncSession, target_date: date = None) -> Dict[str, Any]:
    """
    Computes real-time daily safe-to-spend allowance:
    Formula:
        Monthly Income
      - Contractual Fixed Floor (Rent + Loans + Subscriptions + Insurance)
      - Planned Wealth & Goal Savings Target
      = Free Discretionary Pool
      
      Dynamic Daily Allowance = (Remaining Discretionary Pool) / (Remaining Days in Month)
    """
    today = target_date or date.today()
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    day_of_month = today.day
    days_remaining = max(1, days_in_month - day_of_month + 1)
    month_start = today.replace(day=1)

    # 1. Fetch latest life state or fallbacks
    ls_result = await db.execute(
        select(LifeState)
        .where(LifeState.user_id == user_id)
        .order_by(desc(LifeState.snapshot_date))
        .limit(1)
    )
    life_state = ls_result.scalar_one_or_none()

    monthly_income = float(life_state.monthly_income) if life_state and life_state.monthly_income else 120000.0
    
    # 2. Compute Contractual Fixed Floor from Loans and Subscriptions
    # Loans monthly EMI sum
    loans_result = await db.execute(
        select(func.coalesce(func.sum(Loan.monthly_payment), 0))
        .where(and_(Loan.user_id == user_id, Loan.is_active == True))
    )
    loans_monthly = float(loans_result.scalar() or 0.0)

    # Subscriptions monthly sum
    subs_result = await db.execute(
        select(func.coalesce(func.sum(Subscription.amount), 0))
        .where(and_(Subscription.user_id == user_id, Subscription.is_active == True))
    )
    subs_monthly = float(subs_result.scalar() or 0.0)

    # Fixed living baseline (Rent + Utilities + Groceries necessity floor)
    fixed_necessity_floor = float(life_state.fixed_expenses) if life_state and life_state.fixed_expenses else max(30000.0, loans_monthly + subs_monthly + 15000.0)
    fixed_obligations_total = max(loans_monthly + subs_monthly, fixed_necessity_floor)

    # 3. Monthly Goal Targets
    goals_result = await db.execute(
        select(Goal).where(and_(Goal.user_id == user_id, Goal.status == "in_progress"))
    )
    goals = goals_result.scalars().all()
    # Assume 15% to 25% of income target savings if no explicit monthly contribution
    monthly_goal_savings = monthly_income * 0.25 if not goals else sum(
        float(g.target_amount - g.current_amount) / 12.0 for g in goals if g.target_amount > g.current_amount
    )
    monthly_goal_savings = min(monthly_goal_savings, monthly_income * 0.35)

    # 4. Total Discretionary Pool for Month
    discretionary_pool_total = max(0.0, monthly_income - fixed_obligations_total - monthly_goal_savings)
    baseline_daily_allowance = discretionary_pool_total / days_in_month

    # 5. Month-to-date actual discretionary spending
    mtd_spend_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0))
        .where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.DEBIT,
                Transaction.date >= month_start,
                Transaction.date <= today
            )
        )
    )
    mtd_total_spend = float(mtd_spend_result.scalar() or 0.0)
    
    # Approximate discretionary fraction (total spend minus pro-rated fixed obligations)
    pro_rated_fixed = (fixed_obligations_total / days_in_month) * day_of_month
    mtd_discretionary_spend = max(0.0, mtd_total_spend - pro_rated_fixed)

    # 6. Remaining Discretionary Pool & Today's Allowance
    remaining_discretionary_pool = max(0.0, discretionary_pool_total - mtd_discretionary_spend)
    daily_safe_to_spend = remaining_discretionary_pool / days_remaining

    # 7. Today's actual spending so far
    today_spend_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0))
        .where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.DEBIT,
                Transaction.date == today
            )
        )
    )
    today_spent = float(today_spend_result.scalar() or 0.0)
    today_remaining_allowance = max(0.0, daily_safe_to_spend - today_spent)

    # Status determination
    if daily_safe_to_spend >= baseline_daily_allowance * 0.9:
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

    # Actionable advice
    if today_remaining_allowance > 0:
        advice = f"You can comfortably spend up to ₹{today_remaining_allowance:,.0f} today. Your rent, EMIs, and monthly goals are fully protected."
    else:
        advice = f"Daily budget reached for today (₹{today_spent:,.0f} spent). Limit non-essential expenses until tomorrow morning."

    return {
        "today_date": str(today),
        "days_in_month": days_in_month,
        "day_of_month": day_of_month,
        "days_remaining": days_remaining,
        "monthly_income": monthly_income,
        "fixed_obligations_total": fixed_obligations_total,
        "monthly_goal_savings": monthly_goal_savings,
        "discretionary_pool_total": round(discretionary_pool_total, 2),
        "baseline_daily_allowance": round(baseline_daily_allowance, 2),
        "mtd_discretionary_spend": round(mtd_discretionary_spend, 2),
        "remaining_discretionary_pool": round(remaining_discretionary_pool, 2),
        "daily_safe_to_spend": round(daily_safe_to_spend, 2),
        "today_spent": round(today_spent, 2),
        "today_remaining_allowance": round(today_remaining_allowance, 2),
        "status_code": status_code,
        "status_label": status_label,
        "status_color": color,
        "advice": advice,
    }

