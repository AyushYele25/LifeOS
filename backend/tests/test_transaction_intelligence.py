"""
Tests for Phase 3 — Transaction Intelligence:
- Hybrid ML Classifier (rules + TF-IDF Logistic Regression)
- CSV Import Preview & Execution with duplicate prevention
- User Category Correction Workflow
- Evaluation Metrics Endpoint
"""

import io
import pytest
from uuid import UUID
from datetime import date
from sqlalchemy import select
from app.models import Transaction, Category, AuditLog
from app.ml.classifier import get_classifier


def test_classifier_accuracy_on_known_patterns():
    """Verify hybrid classifier accurately categorizes prominent Indian patterns."""
    classifier = get_classifier()

    samples = [
        ("ACH CREDIT — HDFC Bank CORPORATE SALARY", "Salary"),
        ("UPI/HOUSE RENT/Rahul/LANDLORD", "Rent"),
        ("POS/E-COM RECURRING — NETFLIX INDIA", "Subscriptions"),
        ("UPI/PAYMENT TO SWIGGY/REF-12345", "Food & Dining"),
        ("UPI/PAYMENT TO BLINKIT/REF-56789", "Groceries"),
        ("UPI/PAYMENT TO UBER INDIA/REF-99887", "Transportation"),
        ("BBPS ELECTRICITY BILL — Tata Power", "Electricity"),
        ("POS/PETROL — INDIAN OIL CORPORATION", "Fuel"),
    ]

    for desc, expected_cat in samples:
        cat, conf, is_confident = classifier.predict(desc)
        assert cat == expected_cat
        assert conf >= 0.70
        assert is_confident is True


@pytest.mark.asyncio
async def test_csv_import_preview_endpoint(client):
    """Test previewing a CSV file before importing."""
    # Register user
    reg = await client.post("/api/auth/register", json={
        "email": "preview_user@example.com",
        "password": "Password123!",
        "full_name": "Preview Tester",
    })
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    csv_content = (
        "Date,Description,Amount,Type\n"
        "2026-10-01,ACH CREDIT — CORPORATE SALARY,75000,credit\n"
        "2026-10-03,UPI/HOUSE RENT/LANDLORD,18000,debit\n"
        "2026-10-05,UPI/PAYMENT TO BLINKIT,1250,debit\n"
    )

    files = {"file": ("statement.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = await client.post("/api/transactions/import/preview", headers=headers, files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 3
    assert len(data["preview_sample"]) == 3
    assert data["preview_sample"][0]["predicted_category"] == "Salary"
    assert data["preview_sample"][1]["predicted_category"] == "Rent"
    assert data["preview_sample"][2]["predicted_category"] == "Groceries"


@pytest.mark.asyncio
async def test_csv_import_with_ml_and_duplicate_prevention(client, db_session):
    """Test full CSV import with ML classification and duplicate prevention."""
    reg = await client.post("/api/auth/register", json={
        "email": "import_user@example.com",
        "password": "Password123!",
        "full_name": "Import Tester",
    })
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    csv_content = (
        "Date,Description,Amount,Type\n"
        "2026-09-01,ACH CREDIT — CORPORATE SALARY,85000,credit\n"
        "2026-09-02,UPI/PAYMENT TO SWIGGY,450,debit\n"
        "2026-09-05,BBPS ELECTRICITY BILL — BESCOM,1850,debit\n"
    )

    # 1. First Import
    files = {"file": ("statement.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res1 = await client.post("/api/transactions/import/csv", headers=headers, files=files)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["imported"] == 3
    assert data1["duplicates_skipped"] == 0

    # Verify transactions in DB have ML classification
    list_res = await client.get("/api/transactions", headers=headers)
    txns = list_res.json()["items"]
    assert len(txns) == 3
    for t in txns:
        assert t["category_ml_confidence"] is not None
        assert t["category_ml_confidence"] >= 0.70
        assert t["is_classified_by_user"] is False

    # 2. Second Import with same content -> MUST SKIP DUPLICATES
    files2 = {"file": ("statement.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res2 = await client.post("/api/transactions/import/csv", headers=headers, files=files2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["imported"] == 0
    assert data2["duplicates_skipped"] == 3

    # Total transactions must still be exactly 3
    list_res2 = await client.get("/api/transactions", headers=headers)
    assert list_res2.json()["total"] == 3


@pytest.mark.asyncio
async def test_category_correction_workflow(client, db_session):
    """Test user correcting a transaction category and recording an audit trail."""
    reg = await client.post("/api/auth/register", json={
        "email": "correct_user@example.com",
        "password": "Password123!",
        "full_name": "Correction Tester",
    })
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # Create transaction
    txn_res = await client.post("/api/transactions", headers=headers, json={
        "date": str(date.today()),
        "description": "UPI/GENERAL STORE/REF-123",
        "amount": 500.0,
        "transaction_type": "debit"
    })
    txn_id = txn_res.json()["id"]

    # Fetch Groceries category ID from DB
    cat_res = await db_session.execute(select(Category).where(Category.name == "Groceries").limit(1))
    groceries_cat = cat_res.scalar_one()

    # User corrects category
    patch_res = await client.patch(
        f"/api/transactions/{txn_id}/category",
        headers=headers,
        json={"category_id": str(groceries_cat.id), "notes": "Handmade snack purchase"}
    )
    assert patch_res.status_code == 200
    updated_txn = patch_res.json()
    assert updated_txn["category_id"] == str(groceries_cat.id)
    assert updated_txn["is_classified_by_user"] is True
    assert updated_txn["notes"] == "Handmade snack purchase"

    # Verify audit log was recorded
    audit_res = await db_session.execute(
        select(AuditLog).where(AuditLog.entity_id == str(txn_id))
    )
    audit = audit_res.scalar_one_or_none()
    assert audit is not None
    assert audit.action == "category_correction"
    assert audit.new_values["category_id"] == str(groceries_cat.id)


@pytest.mark.asyncio
async def test_classification_metrics_endpoint(client):
    """Test endpoint returning measured ML classification metrics."""
    reg = await client.post("/api/auth/register", json={
        "email": "metrics_user@example.com",
        "password": "Password123!",
        "full_name": "Metrics Tester",
    })
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    response = await client.get("/api/transactions/classification/metrics", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert data["accuracy"] > 0.85
    assert "macro_f1" in data
    assert "weighted_f1" in data
    assert "per_category" in data
