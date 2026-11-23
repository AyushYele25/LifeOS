"""
User Data Isolation Tests.

Ensures strict multi-tenant boundary: User B can never read, modify,
or access data belonging to User A.
"""

import pytest
from datetime import date


@pytest.mark.asyncio
async def test_user_data_isolation(client):
    # 1. Register User A
    res_a = await client.post("/api/auth/register", json={
        "email": "user_a@example.com",
        "password": "Password123!",
        "full_name": "User Alpha",
    })
    token_a = res_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Register User B
    res_b = await client.post("/api/auth/register", json={
        "email": "user_b@example.com",
        "password": "Password123!",
        "full_name": "User Beta",
    })
    token_b = res_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 3. User A creates an asset
    asset_res = await client.post("/api/assets", headers=headers_a, json={
        "name": "MacBook Pro",
        "asset_type": "electronics",
        "brand": "Apple",
        "model": "M3 Max",
        "purchase_price": 250000.0,
        "current_valuation": 220000.0,
    })
    assert asset_res.status_code == 201
    asset_a_id = asset_res.json()["id"]

    # 4. User A creates a transaction
    txn_res = await client.post("/api/transactions", headers=headers_a, json={
        "date": str(date.today()),
        "description": "Salary Deposit",
        "amount": 150000.0,
        "transaction_type": "credit",
    })
    assert txn_res.status_code == 201
    txn_a_id = txn_res.json()["id"]

    # 5. User B lists assets -> MUST BE EMPTY
    b_assets = await client.get("/api/assets", headers=headers_b)
    assert b_assets.status_code == 200
    assert len(b_assets.json()) == 0

    # 6. User B lists transactions -> MUST BE EMPTY
    b_txns = await client.get("/api/transactions", headers=headers_b)
    assert b_txns.status_code == 200
    assert b_txns.json()["total"] == 0
    assert len(b_txns.json()["items"]) == 0

    # 7. User B tries to directly fetch User A's asset -> MUST RETURN 404
    b_direct_asset = await client.get(f"/api/assets/{asset_a_id}", headers=headers_b)
    assert b_direct_asset.status_code == 404

    # 8. User B tries to delete User A's asset -> MUST RETURN 404
    b_del_asset = await client.delete(f"/api/assets/{asset_a_id}", headers=headers_b)
    assert b_del_asset.status_code == 404

    # 9. User B tries to directly fetch User A's transaction -> MUST RETURN 404
    b_direct_txn = await client.get(f"/api/transactions/{txn_a_id}", headers=headers_b)
    assert b_direct_txn.status_code == 404

    # 10. User A can still retrieve their own asset
    a_own_asset = await client.get(f"/api/assets/{asset_a_id}", headers=headers_a)
    assert a_own_asset.status_code == 200
    assert a_own_asset.json()["name"] == "MacBook Pro"
