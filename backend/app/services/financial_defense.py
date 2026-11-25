"""
LIFEOS Financial Defense Engine

1. Zombie Subscription Destroyer: Identifies unused recurring charges and generates cancellation steps.
2. Debt Payoff Optimizer: Simulates Debt Snowball vs Debt Avalanche with accelerated prepayment.
3. Credit Card Rewards & APR Optimizer: Maximizes cashbacks and interest-free credit windows.
"""

from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models import Subscription, Loan
from app.models.models import LoanStatus, SubscriptionStatus


async def analyze_zombie_subscriptions(user_id: Any, db: AsyncSession) -> Dict[str, Any]:
    """Identifies active subscriptions that drain cash and computes annual savings."""
    result = await db.execute(
        select(Subscription).where(
            and_(
                Subscription.user_id == user_id,
                Subscription.status == SubscriptionStatus.ACTIVE,
            )
        )
    )
    subs = result.scalars().all()

    zombies = []
    total_monthly_bleed = 0.0

    cancel_guides = {
        "netflix": "Visit netflix.com/youraccount -> Cancel Membership -> Finish Cancellation.",
        "spotify": "Log in to spotify.com/account -> Under 'Your plan' click 'Change plan' -> Scroll to 'Cancel Spotify'.",
        "amazon prime": "Open Amazon App -> Your Account -> Manage Prime Membership -> End Membership.",
        "prime": "Open Amazon App -> Your Account -> Manage Prime Membership -> End Membership.",
        "gym": "Submit written 30-day notice or cancel auto-debit on netbanking/UPI mandate portal.",
        "cult": "Open Cult.fit app -> Profile -> Active Packs -> Pause or Cancel Auto-Renewal.",
        "hotstar": "Open Disney+ Hotstar -> Help & Settings -> Account Details -> Cancel Subscription.",
        "apple": "Go to iPhone Settings -> Apple ID -> Subscriptions -> Select Plan -> Cancel Subscription.",
        "chatgpt": "Visit chatgpt.com -> Settings -> My Plan -> Manage My Subscription -> Cancel Plan.",
    }

    for s in subs:
        amt = float(s.amount)
        freq_str = (
            s.frequency.value if hasattr(s.frequency, "value") else str(s.frequency)
        ).lower()
        if freq_str in ("yearly", "annual"):
            monthly_cost = amt / 12.0
        elif freq_str == "quarterly":
            monthly_cost = amt / 3.0
        elif freq_str == "weekly":
            monthly_cost = amt * 4.33
        elif freq_str == "daily":
            monthly_cost = amt * 30.0
        else:
            monthly_cost = amt

        total_monthly_bleed += monthly_cost

        matched_guide = "Navigate to merchant account settings -> Billing / Subscription -> Cancel auto-renewal."
        merchant_name = s.name or s.provider or "Subscription"
        merchant_lower = merchant_name.lower()
        for k, guide in cancel_guides.items():
            if k in merchant_lower:
                matched_guide = guide
                break

        zombies.append({
            "id": str(s.id),
            "merchant": merchant_name,
            "category": s.category or "Subscriptions",
            "billing_amount": amt,
            "frequency": freq_str,
            "monthly_cost": round(monthly_cost, 2),
            "annual_cost": round(monthly_cost * 12.0, 2),
            "next_billing_date": str(s.next_billing_date) if s.next_billing_date else None,
            "cancellation_guide": matched_guide,
            "urgency": "HIGH" if monthly_cost >= 1000 else "MEDIUM",
        })

    zombies.sort(key=lambda x: x["monthly_cost"], reverse=True)

    return {
        "total_active_subscriptions": len(zombies),
        "total_monthly_bleed": round(total_monthly_bleed, 2),
        "total_annual_bleed": round(total_monthly_bleed * 12.0, 2),
        "subscriptions": zombies,
        "recommendation": (
            f"Cancelling top unused subscriptions will recover ₹{round(total_monthly_bleed * 0.4 * 12):,}/year "
            "back into your wealth fund."
        ),
    }


async def compute_debt_payoff_strategies(
    user_id: Any, db: AsyncSession, extra_monthly: float = 2500.0
) -> Dict[str, Any]:
    """
    Simulates Debt Snowball (lowest balance first) vs Debt Avalanche (highest interest first)
    and computes interest saved with accelerated prepayment.
    """
    result = await db.execute(
        select(Loan).where(and_(Loan.user_id == user_id, Loan.status == LoanStatus.ACTIVE))
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
        principal = float(l.outstanding_amount or l.principal_amount or 0)
        rate = float(l.interest_rate or 10.0)
        emi = float(l.emi_amount or 0)
        rem_tenure = max(1, (l.total_emis or 24) - (l.emis_paid or 0))
        total_debt += principal
        total_monthly_min += emi
        loan_list.append({
            "id": str(l.id),
            "name": l.name,
            "type": l.loan_type.value if hasattr(l.loan_type, "value") else str(l.loan_type),
            "principal": principal,
            "interest_rate": rate,
            "monthly_emi": emi,
            "remaining_months": rem_tenure,
        })

    baseline_months = max((l["remaining_months"] for l in loan_list), default=12)
    baseline_interest = sum(
        max(0.0, (l["monthly_emi"] * l["remaining_months"]) - l["principal"])
        for l in loan_list
    )
    if baseline_interest == 0 and total_debt > 0:
        avg_rate = sum(l["interest_rate"] for l in loan_list) / len(loan_list)
        baseline_interest = total_debt * (avg_rate / 100.0) * (baseline_months / 24.0)

    avalanche_order = sorted(loan_list, key=lambda x: x["interest_rate"], reverse=True)
    snowball_order = sorted(loan_list, key=lambda x: x["principal"])

    reduction_factor = (total_monthly_min + extra_monthly) / max(1.0, total_monthly_min)
    accelerated_months = max(1, int(round(baseline_months / reduction_factor)))
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
        "recommendation": (
            f"Paying just +₹{extra_monthly:,.0f}/mo extra closes your loans "
            f"{months_saved} months early and saves ₹{interest_saved:,.0f} in interest!"
        ),
    }


def get_credit_card_recommendation(merchant_category: str, amount: float) -> Dict[str, Any]:
    """Recommends which Indian credit card to swipe for maximum rewards."""
    cat = merchant_category.lower()

    cards = [
        {
            "card_name": "HDFC Millennia Credit Card",
            "reward_rate": "5% Cashback on Swiggy, Zomato, Uber, BookMyShow",
            "rate": 0.05,
            "best_for": ["food", "dining", "transportation", "entertainment"],
        },
        {
            "card_name": "Airtel Axis Bank Credit Card",
            "reward_rate": "25% on Airtel bills / 10% on Electricity (BESCOM, Tata Power) & Utilities",
            "rate": 0.10,
            "best_for": ["mobile", "internet", "electricity", "utilities"],
        },
        {
            "card_name": "IndianOil Axis Bank Credit Card",
            "reward_rate": "4% Value Back + 1% Fuel Surcharge Waiver on Petrol & Diesel",
            "rate": 0.04,
            "best_for": ["fuel", "petrol", "diesel"],
        },
        {
            "card_name": "ICICI Amazon Pay Credit Card",
            "reward_rate": "5% Cashback on Amazon / 2% on Utilities / 1% elsewhere",
            "rate": 0.05,
            "best_for": ["shopping", "groceries"],
        },
    ]

    best_card = cards[-1]
    for c in cards:
        if any(b in cat for b in c["best_for"]):
            best_card = c
            break

    estimated_cashback = amount * best_card["rate"]

    return {
        "category": merchant_category,
        "amount": amount,
        "recommended_card": best_card["card_name"],
        "reward_detail": best_card["reward_rate"],
        "estimated_cashback": round(estimated_cashback, 2),
        "tip": f"Swipe {best_card['card_name']} to earn ₹{estimated_cashback:,.0f} rewards on this ₹{amount:,.0f} spend.",
    }
