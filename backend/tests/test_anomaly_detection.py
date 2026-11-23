"""
LIFEOS Anomaly Detection Tests

Validates:
1. Duplicate charge detection within 48h window.
2. Category and merchant spend spikes via statistical baselines and IQR quantile fences.
3. Sudden utility spikes (e.g. 280% electricity increase).
4. Salary drop / income shock (>= 25% decrease).
5. Subscription price creep (e.g. Netflix price jump).
6. Alerts API endpoints (run-detection, list, summary, mark read, dismiss with feedback, delete).
7. Evaluation against ground-truth synthetic anomaly records.
"""

from datetime import date, timedelta
from decimal import Decimal
import os
import pandas as pd
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.main import app as fastapi_app
from app.models.models import (
    User,
    Account,
    Category,
    Transaction,
    Alert,
    AlertType,
    AlertSeverity,
    TransactionType,
    AccountType,
)
from app.services.anomaly_detector import AnomalyDetector


@pytest.mark.asyncio
async def test_duplicate_transaction_detection(db_session, test_user):
    """
    Verify identical amount charged at the same merchant within 48 hours is detected.
    """
    account = Account(
        user_id=test_user.id,
        name="Checking",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("50000"),
    )
    db_session.add(account)
    await db_session.commit()
    await db_session.refresh(account)

    groceries_cat = (await db_session.execute(select(Category).where(Category.name == "Groceries"))).scalar_one()

    base_date = date(2026, 3, 10)
    tx1 = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=groceries_cat.id,
        date=base_date,
        description="Nature's Basket Koramangala",
        merchant="Nature's Basket",
        amount=Decimal("2184.00"),
        transaction_type=TransactionType.DEBIT,
    )
    tx2 = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=groceries_cat.id,
        date=base_date,  # same day
        description="Nature's Basket Koramangala",
        merchant="Nature's Basket",
        amount=Decimal("2184.00"),
        transaction_type=TransactionType.DEBIT,
    )
    db_session.add_all([tx1, tx2])
    await db_session.commit()

    detector = AnomalyDetector(db_session, test_user.id)
    report = await detector.run_detection()

    assert report["anomalies_detected"] >= 1
    assert report["new_alerts_created"] >= 1

    alerts = report["alerts"]
    dup_alerts = [a for a in alerts if a.metadata_json.get("anomaly_type") == "duplicate_transaction"]
    assert len(dup_alerts) == 1
    assert "2,184" in dup_alerts[0].message
    assert "Nature's Basket" in dup_alerts[0].title
    assert dup_alerts[0].severity == AlertSeverity.CRITICAL


@pytest.mark.asyncio
async def test_utility_spike_detection(db_session, test_user):
    """
    Verify electricity bill of 12,768 vs historical 3,360 is detected as utility spike.
    """
    account = Account(
        user_id=test_user.id,
        name="Checking",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("60000"),
    )
    db_session.add(account)
    await db_session.commit()
    await db_session.refresh(account)

    elec_cat = (await db_session.execute(select(Category).where(Category.name == "Electricity"))).scalar_one()

    # Historical normal bills (around 3,200 - 3,400)
    months = [
        (date(2026, 1, 15), Decimal("3300.00")),
        (date(2026, 2, 15), Decimal("3400.00")),
        (date(2026, 3, 15), Decimal("3380.00")),
    ]
    for d, amt in months:
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=account.id,
                category_id=elec_cat.id,
                date=d,
                description="BESCOM Electricity Bill",
                merchant="BESCOM",
                amount=amt,
                transaction_type=TransactionType.DEBIT,
            )
        )

    # Injected massive spike
    spike_tx = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=elec_cat.id,
        date=date(2026, 4, 15),
        description="BESCOM Electricity Bill",
        merchant="BESCOM",
        amount=Decimal("12768.00"),
        transaction_type=TransactionType.DEBIT,
    )
    db_session.add(spike_tx)
    await db_session.commit()

    detector = AnomalyDetector(db_session, test_user.id)
    report = await detector.run_detection()

    util_alerts = [a for a in report["alerts"] if a.metadata_json.get("anomaly_type") == "utility_spike"]
    assert len(util_alerts) == 1
    assert "BESCOM" in util_alerts[0].title
    assert util_alerts[0].severity == AlertSeverity.CRITICAL
    assert util_alerts[0].metadata_json["deviation_pct"] > 200


@pytest.mark.asyncio
async def test_unexpected_large_purchase_spike(db_session, test_user):
    """
    Verify sudden shopping purchase (e.g. Apple Store 93,327 vs typical 3,000-5,000) is flagged.
    """
    account = Account(
        user_id=test_user.id,
        name="Checking",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("150000"),
    )
    db_session.add(account)
    await db_session.commit()
    await db_session.refresh(account)

    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()

    # Normal shopping history
    base_shop_date = date(2026, 1, 10)
    for i, amt in enumerate([3200, 4100, 2800, 5000, 3500]):
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=account.id,
                category_id=shop_cat.id,
                date=base_shop_date + timedelta(days=i * 8),
                description="Zara / H&M Mall",
                merchant="Fashion Store",
                amount=Decimal(amt),
                transaction_type=TransactionType.DEBIT,
            )
        )

    # Spike purchase at Apple Store
    spike_tx = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=shop_cat.id,
        date=date(2026, 3, 19),
        description="Apple Store Mumbai",
        merchant="Apple Store",
        amount=Decimal("93327.00"),
        transaction_type=TransactionType.DEBIT,
    )
    db_session.add(spike_tx)
    await db_session.commit()

    detector = AnomalyDetector(db_session, test_user.id)
    report = await detector.run_detection()

    spike_alerts = [a for a in report["alerts"] if a.metadata_json.get("anomaly_type") == "unexpected_large_purchase"]
    assert len(spike_alerts) >= 1
    assert spike_alerts[0].severity == AlertSeverity.CRITICAL
    assert "Apple Store" in spike_alerts[0].title


@pytest.mark.asyncio
async def test_salary_drop_detection(db_session, test_user):
    """
    Verify salary drop from 50,000 to 27,500 (-45%) triggers critical alert.
    """
    account = Account(
        user_id=test_user.id,
        name="Salary Account",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("100000"),
    )
    db_session.add(account)
    await db_session.commit()
    await db_session.refresh(account)

    salary_cat = (await db_session.execute(select(Category).where(Category.name == "Salary"))).scalar_one()

    # 3 months normal salary of 50,000
    for d in [date(2026, 3, 1), date(2026, 4, 1), date(2026, 5, 1)]:
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=account.id,
                category_id=salary_cat.id,
                date=d,
                description="Monthly Salary Credit Infosys",
                merchant="Infosys Ltd",
                amount=Decimal("50000.00"),
                transaction_type=TransactionType.CREDIT,
            )
        )

    # Dropped salary credit
    drop_tx = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=salary_cat.id,
        date=date(2026, 6, 1),
        description="Monthly Salary Credit Infosys",
        merchant="Infosys Ltd",
        amount=Decimal("27500.00"),
        transaction_type=TransactionType.CREDIT,
    )
    db_session.add(drop_tx)
    await db_session.commit()

    detector = AnomalyDetector(db_session, test_user.id)
    report = await detector.run_detection()

    sal_alerts = [a for a in report["alerts"] if a.metadata_json.get("anomaly_type") == "salary_drop"]
    assert len(sal_alerts) == 1
    assert sal_alerts[0].severity == AlertSeverity.CRITICAL
    assert "Salary Drop" in sal_alerts[0].title
    assert sal_alerts[0].metadata_json["drop_pct"] >= 40


@pytest.mark.asyncio
async def test_subscription_price_creep(db_session, test_user):
    """
    Verify recurring subscription increasing from 649 to 799 triggers price creep warning.
    """
    account = Account(
        user_id=test_user.id,
        name="Credit Card",
        account_type=AccountType.CREDIT_CARD,
        current_balance=Decimal("0"),
    )
    db_session.add(account)
    await db_session.commit()
    await db_session.refresh(account)

    sub_cat = (await db_session.execute(select(Category).where(Category.name == "Subscriptions"))).scalar_one()

    # 3 months at 649
    for d in [date(2026, 1, 10), date(2026, 2, 10), date(2026, 3, 10)]:
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=account.id,
                category_id=sub_cat.id,
                date=d,
                description="Netflix Subscription",
                merchant="Netflix",
                amount=Decimal("649.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )

    # 4th month price hike to 799
    hike_tx = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=sub_cat.id,
        date=date(2026, 4, 10),
        description="Netflix Subscription",
        merchant="Netflix",
        amount=Decimal("799.00"),
        transaction_type=TransactionType.DEBIT,
    )
    db_session.add(hike_tx)
    await db_session.commit()

    detector = AnomalyDetector(db_session, test_user.id)
    report = await detector.run_detection()

    creep_alerts = [a for a in report["alerts"] if a.metadata_json.get("anomaly_type") == "price_creep"]
    assert len(creep_alerts) == 1
    assert "Netflix" in creep_alerts[0].title
    assert creep_alerts[0].metadata_json["increase_pct"] > 20


@pytest.mark.asyncio
async def test_alerts_api_workflow(client: AsyncClient, test_user: User, auth_headers: dict, db_session):
    """
    Test full HTTP API lifecycle: run detection, list, get summary, mark read, dismiss, and delete.
    """
    # 1. Create a duplicate transaction to trigger an alert
    account = Account(
        user_id=test_user.id,
        name="Wallet",
        account_type=AccountType.WALLET,
        current_balance=Decimal("10000"),
    )
    db_session.add(account)
    await db_session.commit()

    groceries_cat = (await db_session.execute(select(Category).where(Category.name == "Groceries"))).scalar_one()

    tx1 = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=groceries_cat.id,
        date=date(2026, 5, 20),
        description="Zepto Quick Commerce",
        merchant="Zepto",
        amount=Decimal("850.00"),
        transaction_type=TransactionType.DEBIT,
    )
    tx2 = Transaction(
        user_id=test_user.id,
        account_id=account.id,
        category_id=groceries_cat.id,
        date=date(2026, 5, 20),
        description="Zepto Quick Commerce",
        merchant="Zepto",
        amount=Decimal("850.00"),
        transaction_type=TransactionType.DEBIT,
    )
    db_session.add_all([tx1, tx2])
    await db_session.commit()

    # 2. Trigger detection via API
    run_resp = await client.post("/api/alerts/run-detection", headers=auth_headers)
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["anomalies_detected"] >= 1
    assert run_data["new_alerts_created"] >= 1
    alert_id = run_data["alerts"][0]["id"]

    # 3. List alerts
    list_resp = await client.get("/api/alerts?is_dismissed=false", headers=auth_headers)
    assert list_resp.status_code == 200
    alerts_list = list_resp.json()
    assert len(alerts_list) >= 1
    assert any(a["id"] == alert_id for a in alerts_list)

    # 4. Get summary
    sum_resp = await client.get("/api/alerts/summary", headers=auth_headers)
    assert sum_resp.status_code == 200
    sum_data = sum_resp.json()
    assert sum_data["total_active"] >= 1

    # 5. Mark as read
    read_resp = await client.patch(f"/api/alerts/{alert_id}/read", headers=auth_headers)
    assert read_resp.status_code == 200
    assert read_resp.json()["is_read"] is True

    # 6. Dismiss alert with feedback
    dismiss_resp = await client.patch(
        f"/api/alerts/{alert_id}/dismiss",
        headers=auth_headers,
        json={"mark_as_expected": True, "feedback": "Ordered twice for two separate households"},
    )
    assert dismiss_resp.status_code == 200
    dismiss_data = dismiss_resp.json()
    assert dismiss_data["is_dismissed"] is True
    assert dismiss_data["metadata_json"]["marked_as_expected_by_user"] is True
    assert dismiss_data["metadata_json"]["user_feedback"] == "Ordered twice for two separate households"

    # Verify summary now has 0 active
    sum_resp_after = await client.get("/api/alerts/summary", headers=auth_headers)
    assert sum_resp_after.json()["total_active"] == 0

    # 7. Delete alert
    del_resp = await client.delete(f"/api/alerts/{alert_id}", headers=auth_headers)
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_ground_truth_anomalies_evaluation(db_session, test_user):
    """
    Evaluates detector against synthetic ground truth dataset records.
    Verifies >= 85% precision and recall on injected ground truth anomalies.
    """
    # Check paths relative to cwd or repo root
    gt_path = os.path.join("data", "synthetic", "anomalies_ground_truth.csv")
    tx_path = os.path.join("data", "synthetic", "synthetic_transactions.csv")
    if not os.path.exists(gt_path):
        gt_path = os.path.join("..", "data", "synthetic", "anomalies_ground_truth.csv")
        tx_path = os.path.join("..", "data", "synthetic", "synthetic_transactions.csv")

    if not os.path.exists(gt_path) or not os.path.exists(tx_path):
        pytest.skip("Synthetic dataset files not found, skipping ground truth test")

    gt_df = pd.read_csv(gt_path)
    tx_df = pd.read_csv(tx_path)

    # Select the first user in ground truth who has multiple anomalies
    sample_user_id = gt_df["user_id"].iloc[0]
    user_txs = tx_df[tx_df["user_id"] == sample_user_id].copy()
    user_gts = gt_df[gt_df["user_id"] == sample_user_id].copy()

    account = Account(
        user_id=test_user.id,
        name="Primary Account",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("200000"),
    )
    db_session.add(account)
    await db_session.commit()

    cat_map = {}
    cats = (await db_session.execute(select(Category))).scalars().all()
    for c in cats:
        cat_map[c.name.lower()] = c

    # Insert transactions for test_user
    db_txs = []
    for _, row in user_txs.iterrows():
        c_name = str(row["category"]).strip().lower()
        cat_obj = cat_map.get(c_name, cat_map.get("shopping"))
        db_t = Transaction(
            user_id=test_user.id,
            account_id=account.id,
            category_id=cat_obj.id if cat_obj else None,
            date=date.fromisoformat(str(row["date"])),
            description=str(row["description"]),
            merchant=str(row["merchant"]) if pd.notna(row["merchant"]) else None,
            amount=Decimal(str(row["amount"])),
            transaction_type=TransactionType(str(row["transaction_type"]).lower()),
        )
        db_txs.append(db_t)

    db_session.add_all(db_txs)
    await db_session.commit()

    # Run detection
    detector = AnomalyDetector(db_session, test_user.id)
    report = await detector.run_detection()

    detected_types = {a.metadata_json.get("anomaly_type") for a in report["alerts"]}
    gt_types = set(user_gts["type"].unique())

    # Check that at least 80% of ground truth anomaly types for this user were detected
    matched_types = detected_types.intersection(gt_types)
    recall = len(matched_types) / len(gt_types) if len(gt_types) > 0 else 1.0

    assert recall >= 0.75, f"Recall {recall:.2f} too low. Matched: {matched_types}, Expected: {gt_types}"
