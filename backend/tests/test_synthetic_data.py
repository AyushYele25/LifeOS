"""
Tests for Synthetic Data Generator and Anomaly Labeling.
"""

import pytest
from app.services.synthetic_data import generate_synthetic_dataset, seed_database_from_synthetic
from sqlalchemy import select, func
from app.models import User, Transaction, Account, Event


def test_synthetic_data_generation_reproducibility():
    """Verify that same seed produces identical dataset."""
    data_1 = generate_synthetic_dataset(num_users=3, months=2, seed=123)
    data_2 = generate_synthetic_dataset(num_users=3, months=2, seed=123)

    assert len(data_1["users"]) == 3
    assert len(data_1["transactions"]) == len(data_2["transactions"])
    assert data_1["users"][0]["email"] == data_2["users"][0]["email"]
    assert data_1["transactions"][0]["amount"] == data_2["transactions"][0]["amount"]


def test_synthetic_data_anomaly_ground_truth():
    """Verify that injected anomalies have ground-truth metadata."""
    data = generate_synthetic_dataset(num_users=10, months=12, seed=42)
    anomalies = data["anomalies_ground_truth"]

    assert len(anomalies) > 0
    for a in anomalies:
        assert "user_id" in a
        assert "type" in a
        assert a["type"] in ["utility_spike", "salary_drop", "duplicate_transaction", "unexpected_large_purchase"]
        assert "expected" in a
        assert "observed" in a
        assert len(a["evidence"]) > 10


def test_synthetic_data_volume_constraints():
    """Verify that full dataset generates at least 10,000 transactions."""
    data = generate_synthetic_dataset(num_users=50, months=12, seed=42)
    assert len(data["transactions"]) >= 10000


@pytest.mark.asyncio
async def test_synthetic_data_seeding(db_session):
    """Verify that dataset can be seeded into database session."""
    data = generate_synthetic_dataset(num_users=2, months=2, seed=42)
    await seed_database_from_synthetic(data, session=db_session)

    user_count = await db_session.execute(select(func.count(User.id)))
    assert user_count.scalar() >= 2

    txn_count = await db_session.execute(select(func.count(Transaction.id)))
    assert txn_count.scalar() >= 50
