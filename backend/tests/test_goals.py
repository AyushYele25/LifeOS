"""
Tests for Goals API endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_goals_lifecycle_and_crud(client: AsyncClient, auth_headers: dict):
    # 1. Create a goal
    goal_payload = {
        "name": "Emergency Fund 6 Months",
        "description": "Six months of living expenses reserve",
        "target_amount": 300000.00,
        "current_amount": 50000.00,
        "priority": 1,
    }
    resp = await client.post("/api/goals", json=goal_payload, headers=auth_headers)
    assert resp.status_code == 201
    created_goal = resp.json()
    assert created_goal["name"] == "Emergency Fund 6 Months"
    assert float(created_goal["target_amount"]) == 300000.0
    assert float(created_goal["current_amount"]) == 50000.0
    assert created_goal["status"] == "active"
    goal_id = created_goal["id"]

    # 2. List goals
    resp = await client.get("/api/goals", headers=auth_headers)
    assert resp.status_code == 200
    goals = resp.json()
    assert len(goals) >= 1
    assert any(g["id"] == goal_id for g in goals)

    # 3. Get specific goal
    resp = await client.get(f"/api/goals/{goal_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == goal_id

    # 4. Update goal progress
    patch_payload = {
        "current_amount": 300000.00,
    }
    resp = await client.patch(f"/api/goals/{goal_id}", json=patch_payload, headers=auth_headers)
    assert resp.status_code == 200
    updated_goal = resp.json()
    assert float(updated_goal["current_amount"]) == 300000.0
    # Auto-completed since target_amount reached
    assert updated_goal["status"] == "completed"

    # 5. Delete goal
    resp = await client.delete(f"/api/goals/{goal_id}", headers=auth_headers)
    assert resp.status_code == 204

    # 6. Verify 404 after deletion
    resp = await client.get(f"/api/goals/{goal_id}", headers=auth_headers)
    assert resp.status_code == 404
