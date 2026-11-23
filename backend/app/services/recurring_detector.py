"""
LIFEOS Recurring Payment Detection Engine.

Analyzes transaction history using:
- Merchant string normalization
- Interval cadence clustering (weekly, monthly, quarterly, annual)
- Amount variance (handling fixed subscriptions vs variable utility bills)
- Evidence generation and confidence scoring
"""

import re
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from app.models import Transaction, RecurringPattern
from app.models.models import SubscriptionFrequency


def normalize_merchant_name(raw_merchant: Optional[str], description: str) -> str:
    """Extract and normalize canonical merchant name."""
    target = (raw_merchant or description or "").strip()
    lower = target.lower()

    # Brand and category aliases from full string
    if "netflix" in lower:
        return "Netflix"
    if "spotify" in lower:
        return "Spotify"
    if "prime" in lower or "amazon prime" in lower:
        return "Amazon Prime"
    if "hotstar" in lower:
        return "Disney+ Hotstar"
    if "youtube" in lower:
        return "YouTube Premium"
    if "cult.fit" in lower or "cult fit" in lower:
        return "Cult.fit"
    if "airtel" in lower:
        return "Airtel"
    if "jio" in lower:
        return "Jio"
    if "bescom" in lower:
        return "BESCOM Electricity"
    if "tata power" in lower:
        return "Tata Power"
    if "adani" in lower:
        return "Adani Electricity"
    if "landlord" in lower or "rent" in lower:
        return "House Rent"
    if "swiggy" in lower:
        return "Swiggy"
    if "zomato" in lower:
        return "Zomato"
    if "blinkit" in lower:
        return "Blinkit"
    if "zepto" in lower:
        return "Zepto"
    if "uber" in lower:
        return "Uber"
    if "ola" in lower:
        return "Ola"

    # Common prefixes cleanup
    cleaned = re.sub(r"^(upi/payment to |upi/house rent/|pos/e-com recurring — |pos/|bbps |nach debit — |ach credit — )", "", target, flags=re.IGNORECASE)
    cleaned = re.sub(r"/ref-\d+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if "prime" in lower or "amazon prime" in lower:
        return "Amazon Prime"
    if "hotstar" in lower:
        return "Disney+ Hotstar"
    if "youtube" in lower:
        return "YouTube Premium"
    if "cult.fit" in lower or "cult fit" in lower:
        return "Cult.fit"
    if "airtel" in lower:
        return "Airtel"
    if "jio" in lower:
        return "Jio"
    if "bescom" in lower:
        return "BESCOM Electricity"
    if "tata power" in lower:
        return "Tata Power"
    if "adani" in lower:
        return "Adani Electricity"
    if "landlord" in lower or "rent" in lower:
        return "House Rent"
    if "swiggy" in lower:
        return "Swiggy"
    if "zomato" in lower:
        return "Zomato"
    if "blinkit" in lower:
        return "Blinkit"
    if "zepto" in lower:
        return "Zepto"
    if "uber" in lower:
        return "Uber"
    if "ola" in lower:
        return "Ola"

    return cleaned.title() if len(cleaned) > 2 else "Unknown Merchant"


def classify_cadence(mean_interval: float, std_interval: float) -> Tuple[Optional[SubscriptionFrequency], float]:
    """
    Determine cadence frequency and interval regularity score.
    Returns: (SubscriptionFrequency, interval_score)
    """
    if mean_interval <= 0:
        return None, 0.0

    # Coefficient of variation for interval
    cv = std_interval / mean_interval if mean_interval > 0 else 1.0
    interval_score = max(0.0, 1.0 - cv)

    # Cadence thresholds in days
    if 5 <= mean_interval <= 9:
        return SubscriptionFrequency.WEEKLY, interval_score
    elif 25 <= mean_interval <= 35:
        return SubscriptionFrequency.MONTHLY, interval_score
    elif 80 <= mean_interval <= 100:
        return SubscriptionFrequency.QUARTERLY, interval_score
    elif 170 <= mean_interval <= 195:
        return SubscriptionFrequency.HALF_YEARLY, interval_score
    elif 345 <= mean_interval <= 385:
        return SubscriptionFrequency.YEARLY, interval_score

    return None, 0.0


class RecurringDetector:
    """Engine that detects recurring payments from transaction streams."""

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence

    def analyze_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a list of transaction dicts and return candidate recurring patterns.
        Each txn must have: 'date', 'amount', 'description', optionally 'merchant', 'category'.
        """
        # 1. Group by normalized merchant
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for t in transactions:
            m_name = normalize_merchant_name(t.get("merchant"), t.get("description", ""))
            if m_name not in groups:
                groups[m_name] = []
            groups[m_name].append(t)

        candidates = []

        for merchant, txns in groups.items():
            # Filter: need at least 2 txns for annual, 3 for others
            if len(txns) < 2:
                continue

            # Sort chronologically
            sorted_txns = sorted(txns, key=lambda x: x["date"] if isinstance(x["date"], date) else date.fromisoformat(str(x["date"])))
            dates = [x["date"] if isinstance(x["date"], date) else date.fromisoformat(str(x["date"])) for x in sorted_txns]
            amounts = [float(x["amount"]) for x in sorted_txns]

            # Calculate intervals
            intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
            if not intervals:
                continue

            mean_int = float(np.mean(intervals))
            std_int = float(np.std(intervals)) if len(intervals) > 1 else 0.0

            frequency, interval_score = classify_cadence(mean_int, std_int)
            if frequency is None:
                continue

            # Annual needs at least 2 occurrences; Monthly/Weekly/Quarterly need at least 3
            if frequency != SubscriptionFrequency.YEARLY and len(sorted_txns) < 3:
                continue

            # Amount analysis (handling fixed vs variable)
            mean_amt = float(np.mean(amounts))
            std_amt = float(np.std(amounts)) if len(amounts) > 1 else 0.0
            amt_cv = std_amt / mean_amt if mean_amt > 0 else 1.0

            # Variable bills (like electricity) can have ~20-40% CV; fixed subs have <5% CV
            amount_score = max(0.0, 1.0 - amt_cv)
            # Count factor: more observations increase confidence
            count_factor = min(1.0, len(sorted_txns) / 4.0)

            # Composite confidence score
            confidence = (0.50 * interval_score) + (0.30 * amount_score) + (0.20 * count_factor)
            confidence = round(float(np.clip(confidence, 0.0, 0.99)), 3)

            if confidence < self.min_confidence:
                continue

            last_date = dates[-1]
            next_date = last_date + timedelta(days=int(round(mean_int)))
            category = sorted_txns[-1].get("category") or "Subscriptions"

            evidence = (
                f"Detected {len(sorted_txns)} {frequency.value} payments averaging ₹{mean_amt:.2f} "
                f"(interval: {mean_int:.1f}±{std_int:.1f} days, cadence score: {interval_score*100:.0f}%)"
            )

            candidates.append({
                "merchant": merchant,
                "category": category,
                "average_amount": Decimal(str(round(mean_amt, 2))),
                "frequency": frequency,
                "average_interval_days": round(mean_int, 1),
                "last_transaction_date": last_date,
                "next_expected_date": next_date,
                "confidence": confidence,
                "transaction_count": len(sorted_txns),
                "evidence": evidence,
                "amount_min": Decimal(str(round(min(amounts), 2))),
                "amount_max": Decimal(str(round(max(amounts), 2))),
            })

        return sorted(candidates, key=lambda c: c["confidence"], reverse=True)

    async def detect_and_save_for_user(self, user_id: Any, db: AsyncSession) -> List[RecurringPattern]:
        """Fetch user transactions from DB, detect patterns, and upsert RecurringPattern records."""
        result = await db.execute(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.date)
        )
        txns = result.scalars().all()

        txn_dicts = [
            {
                "date": t.date,
                "amount": float(t.amount),
                "description": t.description,
                "merchant": t.merchant,
                "category": "Miscellaneous",
            }
            for t in txns
        ]

        candidates = self.analyze_transactions(txn_dicts)
        saved_patterns = []

        for c in candidates:
            # Check existing pattern
            existing_res = await db.execute(
                select(RecurringPattern).where(
                    and_(
                        RecurringPattern.user_id == user_id,
                        RecurringPattern.merchant == c["merchant"],
                    )
                ).limit(1)
            )
            pattern = existing_res.scalar_one_or_none()

            if not pattern:
                pattern = RecurringPattern(
                    user_id=user_id,
                    merchant=c["merchant"],
                    category=c["category"],
                    average_amount=c["average_amount"],
                    frequency=c["frequency"],
                    average_interval_days=c["average_interval_days"],
                    last_transaction_date=c["last_transaction_date"],
                    next_expected_date=c["next_expected_date"],
                    confidence=c["confidence"],
                    transaction_count=c["transaction_count"],
                    is_confirmed=False,
                )
                db.add(pattern)
            else:
                pattern.average_amount = c["average_amount"]
                pattern.average_interval_days = c["average_interval_days"]
                pattern.last_transaction_date = c["last_transaction_date"]
                pattern.next_expected_date = c["next_expected_date"]
                pattern.confidence = c["confidence"]
                pattern.transaction_count = c["transaction_count"]

            saved_patterns.append(pattern)

        await db.flush()
        return saved_patterns
