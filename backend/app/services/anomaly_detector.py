"""
LIFEOS Anomaly Detection Service

Hybrid statistical + machine learning anomaly detection engine for financial transactions:
1. Duplicate charge detection (same merchant, identical amount within 48 hours).
2. Category & merchant spend spikes using rolling baselines & IQR upper fences.
3. Salary drop / income shock detection (>= 25% drop).
4. Subscription price creep (e.g., Netflix/broadband price hikes).
5. Cash-drain velocity (rolling 7-day burn rate vs 30-day baseline).
6. Multi-dimensional outlier scoring using scikit-learn Isolation Forest.
"""

from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.models import (
    Transaction,
    Category,
    Alert,
    AlertType,
    AlertSeverity,
    TransactionType,
    AuditLog,
)


class AnomalyDetector:
    """
    Stateful anomaly detection engine for a single user's financial activity.
    """

    def __init__(self, db: AsyncSession, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id

    async def run_detection(self) -> Dict[str, Any]:
        """
        Executes all anomaly detection checks and persists Alert records.
        Returns a summary report with detected anomalies and created alerts.
        """
        # 1. Fetch all transactions for this user ordered chronologically
        query = (
            select(Transaction)
            .options(selectinload(Transaction.category))
            .where(Transaction.user_id == self.user_id)
            .order_by(Transaction.date.asc(), Transaction.id.asc())
        )
        result = await self.db.execute(query)
        transactions = list(result.scalars().all())

        if not transactions:
            return {
                "scanned_transactions": 0,
                "anomalies_detected": 0,
                "new_alerts_created": 0,
                "alerts": [],
            }

        # 2. Fetch existing alerts to prevent duplicate notifications
        alert_query = select(Alert).where(Alert.user_id == self.user_id)
        alert_res = await self.db.execute(alert_query)
        existing_alerts = alert_res.scalars().all()
        existing_alert_keys = set()
        for a in existing_alerts:
            if a.source_id:
                existing_alert_keys.add((str(a.source_id), a.alert_type))
            if a.metadata_json and "anomaly_type" in a.metadata_json and "transaction_id" in a.metadata_json:
                existing_alert_keys.add((str(a.metadata_json["transaction_id"]), a.metadata_json["anomaly_type"]))

        detected_anomalies: List[Dict[str, Any]] = []

        # 3. Rule 1: Duplicate Charge Detection (window: 48 hours)
        duplicate_anomalies = self._detect_duplicates(transactions)
        detected_anomalies.extend(duplicate_anomalies)

        # 4. Rule 2: Spend Spikes (Quantile Fences & Statistical Baselines)
        spend_spike_anomalies = self._detect_spend_spikes(transactions)
        detected_anomalies.extend(spend_spike_anomalies)

        # 5. Rule 3: Salary Drop / Income Shock
        salary_drop_anomalies = self._detect_salary_drops(transactions)
        detected_anomalies.extend(salary_drop_anomalies)

        # 6. Rule 4: Subscription Price Creep
        subscription_anomalies = self._detect_price_creep(transactions)
        detected_anomalies.extend(subscription_anomalies)

        # 7. Rule 5: Cash Drain Velocity
        cash_drain_anomalies = self._detect_cash_drain(transactions)
        detected_anomalies.extend(cash_drain_anomalies)

        # 8. ML: Isolation Forest Multi-dimensional Scoring
        if len(transactions) >= 20:
            self._apply_isolation_forest(transactions)

        # 9. Persist alerts and update transaction flags
        new_alerts: List[Alert] = []
        flagged_tx_ids = set()

        for anom in detected_anomalies:
            tx = anom.get("transaction")
            tx_id = tx.id if tx else None

            if tx:
                tx.is_anomaly = True
                tx.anomaly_score = max(tx.anomaly_score or 0.0, anom.get("score", 0.8))
                flagged_tx_ids.add(tx.id)

            # Check if alert already exists
            key1 = (str(tx_id), AlertType.ANOMALY) if tx_id else None
            key2 = (str(tx_id), anom.get("anomaly_type")) if tx_id else None

            if (key1 and key1 in existing_alert_keys) or (key2 and key2 in existing_alert_keys):
                continue

            alert = Alert(
                user_id=self.user_id,
                alert_type=AlertType.ANOMALY,
                severity=anom["severity"],
                title=anom["title"],
                message=anom["message"],
                source_type="transaction" if tx_id else "cash_drain",
                source_id=tx_id,
                metadata_json=anom.get("metadata", {}),
            )
            self.db.add(alert)
            new_alerts.append(alert)

            if key1:
                existing_alert_keys.add(key1)
            if key2:
                existing_alert_keys.add(key2)

        # 10. Audit log
        audit = AuditLog(
            user_id=self.user_id,
            action="anomaly_detection_run",
            entity_type="alert",
            details={
                "scanned_transactions": len(transactions),
                "anomalies_detected": len(detected_anomalies),
                "new_alerts_created": len(new_alerts),
            },
        )
        self.db.add(audit)
        await self.db.commit()

        for a in new_alerts:
            await self.db.refresh(a)

        return {
            "scanned_transactions": len(transactions),
            "anomalies_detected": len(detected_anomalies),
            "new_alerts_created": len(new_alerts),
            "alerts": new_alerts,
        }

    def _detect_duplicates(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Flags identical debit amounts at the same merchant within a 48-hour window.
        """
        anomalies = []
        debits = [
            t for t in transactions
            if t.transaction_type == TransactionType.DEBIT and t.amount and t.amount > Decimal("50")
        ]

        for i in range(len(debits)):
            t1 = debits[i]
            merchant_1 = (t1.merchant or t1.description or "").strip().lower()
            if not merchant_1:
                continue

            for j in range(i + 1, len(debits)):
                t2 = debits[j]
                merchant_2 = (t2.merchant or t2.description or "").strip().lower()

                # Check date delta
                date_diff = (t2.date - t1.date).days
                if date_diff > 2:
                    break  # Chronologically ordered, further ones will be even later

                # Check merchant equivalence (exact or substring match)
                if merchant_1 == merchant_2 or merchant_1 in merchant_2 or merchant_2 in merchant_1:
                    # Check identical amount (within 0.05 tolerance)
                    if abs(t1.amount - t2.amount) <= Decimal("0.05"):
                        severity = AlertSeverity.CRITICAL if t2.amount >= Decimal("2000") else AlertSeverity.WARNING
                        time_desc = "the same day" if date_diff == 0 else f"{date_diff} day(s) apart"
                        anomalies.append({
                            "transaction": t2,
                            "anomaly_type": "duplicate_transaction",
                            "severity": severity,
                            "score": 0.95,
                            "title": f"Duplicate Charge: {t2.merchant or t2.description}",
                            "message": (
                                f"Identical amount ₹{t2.amount:,.2f} charged twice at "
                                f"{t2.merchant or t2.description} on {t2.date} ({time_desc})."
                            ),
                            "metadata": {
                                "anomaly_type": "duplicate_transaction",
                                "transaction_id": str(t2.id),
                                "original_transaction_id": str(t1.id),
                                "merchant": t2.merchant,
                                "amount": float(t2.amount),
                                "date": str(t2.date),
                                "days_apart": date_diff,
                            },
                        })
        return anomalies

    def _detect_spend_spikes(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Statistical baseline detection:
        - Category-based IQR quantile upper fences: Q75 + 2.5 * IQR.
        - Merchant-based spikes for established merchants (e.g. utilities, groceries).
        """
        anomalies = []
        category_history: Dict[str, List[Decimal]] = {}
        merchant_history: Dict[str, List[Decimal]] = {}

        for t in transactions:
            if t.transaction_type != TransactionType.DEBIT or not t.amount:
                continue

            cat_name = t.category.name if t.category else "Uncategorized"
            merchant_name = (t.merchant or "").strip()
            amount = t.amount

            # Check Utility spike specifically (e.g. Electricity, Water, Gas)
            is_utility = cat_name.lower() in ["utilities", "electricity", "water", "bills", "broadband"]

            # 1. Check merchant baseline spike if merchant has historical records
            if merchant_name and merchant_name in merchant_history and len(merchant_history[merchant_name]) >= 2:
                hist_m = [float(x) for x in merchant_history[merchant_name]]
                mean_m = np.mean(hist_m)
                median_m = np.median(hist_m)

                if is_utility and float(amount) >= 2.0 * mean_m and float(amount) >= 1000:
                    pct_increase = ((float(amount) - mean_m) / mean_m) * 100
                    anomalies.append({
                        "transaction": t,
                        "anomaly_type": "utility_spike",
                        "severity": AlertSeverity.CRITICAL if pct_increase >= 150 else AlertSeverity.WARNING,
                        "score": min(0.99, 0.70 + (pct_increase / 500.0)),
                        "title": f"Unusual Utility Spike: {merchant_name}",
                        "message": (
                            f"{merchant_name} bill ₹{float(amount):,.2f} is {pct_increase:.0f}% higher "
                            f"than historical average of ₹{mean_m:,.2f}."
                        ),
                        "metadata": {
                            "anomaly_type": "utility_spike",
                            "transaction_id": str(t.id),
                            "merchant": merchant_name,
                            "category": cat_name,
                            "observed": float(amount),
                            "expected": round(mean_m, 2),
                            "deviation_pct": round(pct_increase, 1),
                        },
                    })
                    merchant_history[merchant_name].append(amount)
                    if cat_name not in category_history:
                        category_history[cat_name] = []
                    category_history[cat_name].append(amount)
                    continue

            # 2. Check Category-level Quantile Fences
            if cat_name in category_history and len(category_history[cat_name]) >= 4:
                hist_vals = [float(x) for x in category_history[cat_name]]
                q25, q50, q75 = np.percentile(hist_vals, [25, 50, 75])
                iqr = max(q75 - q25, 100.0)
                upper_fence = q75 + (2.5 * iqr)
                median_val = max(q50, 100.0)

                # Check if this transaction represents an extreme outlier
                if float(amount) > upper_fence and float(amount) >= (2.5 * median_val) and float(amount) >= 2000:
                    pct_over = ((float(amount) - median_val) / median_val) * 100
                    title = (
                        f"Unexpected Large Purchase: {merchant_name or cat_name}"
                        if cat_name.lower() in ["shopping", "electronics", "personal"]
                        else f"Spend Spike: {cat_name}"
                    )
                    anomalies.append({
                        "transaction": t,
                        "anomaly_type": "unexpected_large_purchase",
                        "severity": AlertSeverity.CRITICAL if float(amount) >= 25000 or pct_over >= 300 else AlertSeverity.WARNING,
                        "score": min(0.98, 0.65 + (pct_over / 600.0)),
                        "title": title,
                        "message": (
                            f"Large purchase of ₹{float(amount):,.2f} at {merchant_name or cat_name} "
                            f"is {pct_over:.0f}% higher than typical {cat_name} spend."
                        ),
                        "metadata": {
                            "anomaly_type": "unexpected_large_purchase",
                            "transaction_id": str(t.id),
                            "merchant": merchant_name,
                            "category": cat_name,
                            "observed": float(amount),
                            "expected": round(median_val, 2),
                            "upper_fence": round(upper_fence, 2),
                            "deviation_pct": round(pct_over, 1),
                        },
                    })

            # Record history
            if cat_name not in category_history:
                category_history[cat_name] = []
            category_history[cat_name].append(amount)

            if merchant_name:
                if merchant_name not in merchant_history:
                    merchant_history[merchant_name] = []
                merchant_history[merchant_name].append(amount)

        return anomalies

    def _detect_salary_drops(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Detects sudden unexpected salary decreases (>= 25% drop).
        """
        anomalies = []
        salary_history: List[Tuple[date, Decimal, Transaction]] = []

        for t in transactions:
            if t.transaction_type != TransactionType.CREDIT:
                continue

            desc_lower = (t.description or "").lower()
            cat_name = (t.category.name if t.category else "").lower()
            is_salary = "salary" in desc_lower or "payroll" in desc_lower or cat_name == "salary"

            if is_salary and t.amount and t.amount >= Decimal("5000"):
                if salary_history:
                    # Compare with historical median salary
                    hist_amounts = [float(item[1]) for item in salary_history]
                    baseline_salary = np.median(hist_amounts)
                    current_salary = float(t.amount)

                    # If salary dropped by >= 25%
                    if current_salary <= (0.75 * baseline_salary):
                        drop_pct = ((baseline_salary - current_salary) / baseline_salary) * 100
                        anomalies.append({
                            "transaction": t,
                            "anomaly_type": "salary_drop",
                            "severity": AlertSeverity.CRITICAL,
                            "score": 0.96,
                            "title": "Income Shock: Salary Drop Detected",
                            "message": (
                                f"Salary dropped from ₹{baseline_salary:,.2f} to ₹{current_salary:,.2f} "
                                f"(-{drop_pct:.0f}% unexpected decrease) on {t.date}."
                            ),
                            "metadata": {
                                "anomaly_type": "salary_drop",
                                "transaction_id": str(t.id),
                                "expected": round(baseline_salary, 2),
                                "observed": round(current_salary, 2),
                                "drop_pct": round(drop_pct, 1),
                                "date": str(t.date),
                            },
                        })

                salary_history.append((t.date, t.amount, t))

        return anomalies

    def _detect_price_creep(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Detects price increases in recurring subscriptions / regular bills.
        """
        anomalies = []
        recurring_history: Dict[str, List[Tuple[date, Decimal, Transaction]]] = {}

        for t in transactions:
            if t.transaction_type != TransactionType.DEBIT or not t.amount:
                continue

            merchant = (t.merchant or "").strip()
            if not merchant:
                continue

            # Group recurring merchants
            if merchant not in recurring_history:
                recurring_history[merchant] = []

            history = recurring_history[merchant]
            if len(history) >= 2:
                # Calculate previous amount baseline
                prev_amounts = [float(x[1]) for x in history[-4:]]
                median_prev = np.median(prev_amounts)
                curr_amount = float(t.amount)

                # Flag if price increased by > 15% and is at least ₹40 more
                if curr_amount >= (1.15 * median_prev) and (curr_amount - median_prev) >= 40:
                    pct_increase = ((curr_amount - median_prev) / median_prev) * 100
                    anomalies.append({
                        "transaction": t,
                        "anomaly_type": "price_creep",
                        "severity": AlertSeverity.WARNING,
                        "score": 0.85,
                        "title": f"Subscription Price Creep: {merchant}",
                        "message": (
                            f"{merchant} subscription increased from ₹{median_prev:,.2f} to "
                            f"₹{curr_amount:,.2f} (+{pct_increase:.0f}% increase) on {t.date}."
                        ),
                        "metadata": {
                            "anomaly_type": "price_creep",
                            "transaction_id": str(t.id),
                            "merchant": merchant,
                            "expected": round(median_prev, 2),
                            "observed": round(curr_amount, 2),
                            "increase_pct": round(pct_increase, 1),
                        },
                    })

            history.append((t.date, t.amount, t))

        return anomalies

    def _detect_cash_drain(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Flags rapid cash drain where the most recent 7-day discretionary spend
        is >= 2.5x higher than the 30-day weekly baseline pace.
        """
        anomalies = []
        debits = [
            t for t in transactions
            if t.transaction_type == TransactionType.DEBIT and t.amount
        ]
        if len(debits) < 15:
            return anomalies

        latest_date = max(t.date for t in debits)
        seven_days_ago = latest_date - timedelta(days=7)
        thirty_days_ago = latest_date - timedelta(days=30)

        spend_7d = sum(
            float(t.amount) for t in debits
            if seven_days_ago < t.date <= latest_date
        )
        spend_30d = sum(
            float(t.amount) for t in debits
            if thirty_days_ago < t.date <= latest_date
        )

        weekly_baseline = (spend_30d / 30.0) * 7.0

        if weekly_baseline >= 5000 and spend_7d >= (2.5 * weekly_baseline) and spend_7d >= 20000:
            burn_ratio = spend_7d / weekly_baseline
            anomalies.append({
                "transaction": None,
                "anomaly_type": "cash_drain",
                "severity": AlertSeverity.WARNING,
                "score": 0.88,
                "title": "High Cash Burn Rate Alert",
                "message": (
                    f"Accelerated outflow: ₹{spend_7d:,.2f} spent in the last 7 days is "
                    f"{burn_ratio:.1f}x higher than your typical weekly burn rate (₹{weekly_baseline:,.2f})."
                ),
                "metadata": {
                    "anomaly_type": "cash_drain",
                    "seven_day_spend": round(spend_7d, 2),
                    "weekly_baseline": round(weekly_baseline, 2),
                    "burn_ratio": round(burn_ratio, 2),
                    "period_end": str(latest_date),
                },
            })

        return anomalies

    def _apply_isolation_forest(self, transactions: List[Transaction]) -> None:
        """
        Fits an Isolation Forest to score multivariate outlier behavior
        across [amount, day_of_month, day_of_week, is_debit].
        """
        valid_txs = [t for t in transactions if t.amount is not None]
        if len(valid_txs) < 20:
            return

        features = []
        for t in valid_txs:
            amount_log = float(np.log1p(float(t.amount)))
            day_m = float(t.date.day)
            day_w = float(t.date.weekday())
            is_debit = 1.0 if t.transaction_type == TransactionType.DEBIT else 0.0
            features.append([amount_log, day_m, day_w, is_debit])

        X = np.array(features)
        model = IsolationForest(contamination=0.03, random_state=42, n_estimators=50)
        model.fit(X)
        scores = -model.score_samples(X)  # Higher score = more anomalous

        min_s, max_s = float(np.min(scores)), float(np.max(scores))
        score_range = max_s - min_s if max_s > min_s else 1.0

        threshold = np.percentile(scores, 97)  # Top 3%
        for idx, t in enumerate(valid_txs):
            norm_score = float((scores[idx] - min_s) / score_range)
            if scores[idx] >= threshold and norm_score >= 0.75:
                t.is_anomaly = True
                t.anomaly_score = max(t.anomaly_score or 0.0, round(norm_score, 4))
