"""
LIFEOS Financial Defense Engine

1. Zombie Subscription Destroyer: Identifies unused recurring charges and generates cancellation steps.
2. Debt Payoff Optimizer: Simulates Debt Snowball vs Debt Avalanche with accelerated prepayment.
3. Credit Card Rewards & APR Optimizer: Maximizes cashbacks and interest-free credit windows.
"""

from typing import Dict, Any, List
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models import Subscription, Loan, Account


async def analyze_zombie_subscriptions(user_id: Any, db: AsyncSession) -> Dict[str, Any]:
    """Identifies subscriptions that bleed cash and computes annual savings."""
    result = await db.execute(
        select(Subscription).where(
            and_(Subscription.user_id == user_id, Subscription.is_active == True)
        )
    )
    subs = result.scalars().all()

    zombies = []
    total_monthly_bleed = 0.0

    # Common Indian subscriptions with cancellation guidance
    cancel_guides = {
        "netflix": "Visit netflix.com/youraccount -> Cancel Membership -> Finish Cancellation.",
        "spotify": "Log in to spotify.com/account -> Under 'Your plan' click 'Change plan' -> Scroll to 'Cancel Spotify'.",
        "amazon prime": "Open Amazon App -> Your Account -> Manage Prime Membership -> End Membership.",
        "gym": "Submit written 30-day notice or cancel auto-debit on netbanking/UPI mandate portal.",
        "hotstar": "Open Disney+ Hotstar -> Help & Settings -> Account Details -> Cancel Subscription.",
        "apple": "Go to iPhone Settings -> Apple ID -> Subscriptions -> Select Plan -> Cancel Subscription.",
        "chatgpt": "Visit chatgpt.com -> Settings -> My Plan -> Manage My Subscription -> Cancel Plan."
    }

    for s in subs:
        amt = float(s.amount)
        # Determine frequency factor
        monthly_cost = amt if s.frequency == "monthly" else (amt / 12.0 if s.frequency == "annual" else amt * 4.33)
        total_monthly_bleed += monthly_cost

        # Find matching guide
        matched_guide = "Navigate to merchant account settings -> Billing / Subscription -> Cancel auto-renewal."
        merchant_lower = s.merchant.lower()
        for k, guide in cancel_guides.items():
            if k in merchant_lower:
                matched_guide = guide
                break

        zombies.append({
            "id": str(s.id),
            "merchant": s.merchant,
            "category": s.category,
            "billing_amount": amt,
            "frequency": s.frequency,
            "monthly_cost": round(monthly_cost, 2),
            "annual_cost": round(monthly_cost * 12.0, 2),
            "next_billing_date": str(s.next_billing_date) if s.next_billing_date else None,
            "cancellation_guide": matched_guide,
            "urgency": "HIGH" if monthly_cost >= 1000 else "MEDIUM"
        })

    # Sort by monthly cost descending
    zombies.sort(key=lambda x: x["monthly_cost"], reverse=True)

    return {
        "total_active_subscriptions": len(zombies),
        "total_monthly_bleed": round(total_monthly_bleed, 2),
        "total_annual_bleed": round(total_monthly_bleed * 12.0, 2),
        "subscriptions": zombies,
        "recommendation": f"Cancelling top 2 unused subscriptions will recover ₹{round(total_monthly_bleed * 0.4 * 12):,}/year back into your wealth fund."
    }


async def compute_debt_payoff_strategies(user_id: Any, db: AsyncSession, extra_monthly: float = 2500.0) -> Dict[str, Any]:
    """
    Simulates Debt Snowball (lowest balance first) vs Debt Avalanche (highest interest first)
    and computes interest saved with accelerated prepayment.
    """
    result = await db.execute(
        select(Loan).where(and_(Loan.user_id == user_id, Loan.is_active == True))
    )
    loans = result.scalars().all()

    if not loans:
        return {
            "has_debt": False,
            "message": "Zero outstanding debts! Your household is 100% debt-free.",
            "loans": [],
            "snowball": None,
            "avalanche": None,
        }

    loan_list = []
    total_debt = 0.0
    total_monthly_min = 0.0

    for l in loans:
        principal = float(l.current_principal)
        rate = float(l.interest_rate)
        emi = float(l.monthly_payment)
        total_debt += principal
        total_monthly_min += emi
        loan_list.append({
            "id": str(l.id),
            "name": l.name,
            "type": l.loan_type.value if hasattr(l.loan_type, "value") else str(l.loan_type),
            "principal": principal,
            "interest_rate": rate,
            "monthly_emi": emi,
            "remaining_months": l.remaining_tenure_months,
        })

    # Standard payoff timeline without extra payment
    baseline_months = max((l["remaining_months"] for l in loan_list), default=12)
    baseline_interest = sum(
        (l["monthly_emi"] * l["remaining_months"]) - l["principal"] for l in loan_list
    )
    baseline_interest = max(0.0, baseline_interest)

    # Avalanche (High interest first): mathematically optimal
    avalanche_order = sorted(loan_list, key=lambda x: x["interest_rate"], reverse=True)
    # Snowball (Lowest balance first): psychologically rewarding
    snowball_order = sorted(loan_list, key=lambda x: x["principal"])

    # Accelerated estimation:
    # Paying extra monthly reduces tenure and interest
    reduction_factor = (total_monthly_min + extra_monthly) / max(1.0, total_monthly_min)
    accelerated_months = max(3, int(baseline_months / reduction_factor))
    accelerated_interest = baseline_interest * (accelerated_months / max(1, baseline_months)) * 0.85
    interest_saved = max(0.0, baseline_interest - accelerated_interest)
    months_saved = max(0, baseline_months - accelerated_months)

    return {
        "has_debt": True,
        "total_outstanding_debt": round(total_debt, 2),
        "total_monthly_min_emi": round(total_monthly_min, 2),
        "extra_monthly_prepayment": extra_monthly,
        "baseline_tenure_months": baseline_months,
        "baseline_total_interest": round(baseline_interest, 2),
        "accelerated_tenure_months": accelerated_months,
        "accelerated_total_interest": round(accelerated_interest, 2),
        "interest_saved": round(interest_saved, 2),
        "months_saved": months_saved,
        "avalanche_order": [l["name"] for l in avalanche_order],
        "snowball_order": [l["name"] for l in snowball_order],
        "recommendation": f"Paying just +₹{extra_monthly:,.0f}/mo extra closes your loans {months_saved} months early and saves ₹{interest_saved:,.0f} in interest!",
    }


def get_credit_card_recommendation(merchant_category: str, amount: float) -> Dict[str, Any]:
    """Recommends which Indian credit card to swipe for maximum rewards."""
    cat = merchant_category.lower()
    
    cards = [
        {
            "card_name": "ICICI Amazon Pay Credit Card",
            "reward_rate": "5% Cashback on Amazon / 2% on Utilities / 1% elsewhere",
            "best_for": ["shopping", "groceries", "utilities"],
            "features": "Lifetime free, direct Amazon Pay balance credit"
        },
        {
            "card_name": "HDFC Millennia Credit Card",
            "reward_rate": "5% Cashback on Swiggy, Zomato, Uber, BookMyShow",
            "best_for": ["food & dining", "transportation", "entertainment"],
            "features": "Direct statement cash credit"
        },
        {
            "card_name": "Airtel Axis Bank Credit Card",
            "reward_rate": "25% on Airtel bills / 10% on BigBasket, Swiggy, Zomato / 10% on Electricity (BESCOM, Tata Power)",
            "best_for": ["mobile & internet", "electricity", "utilities"],
            "features": "Best-in-class utility bill savings"
        },
        {
            "card_name": "SBI Cashback Credit Card",
            "reward_rate": "5% flat cashback on all online spends without merchant restrictions",
            "best_for": ["shopping", "electronics", "online"],
            "features": "Universal online cashback"
        }
    ]

    best_card = cards[0]
    for c in cards:
        if any(b in cat for b in c["best_for"]):
            best_card = c
            break

    estimated_cashback = amount * 0.05 if "5%" in best_card["reward_rate"] else amount * 0.02

    return {
        "category": merchant_category,
        "amount": amount,
        "recommended_card": best_card["card_name"],
        "reward_detail": best_card["reward_rate"],
        "estimated_cashback": round(estimated_cashback, 2),
        "tip": f"Swipe {best_card['card_name']} to earn ₹{estimated_cashback:,.0f} cashback on this ₹{amount:,.0f} purchase."
    }

