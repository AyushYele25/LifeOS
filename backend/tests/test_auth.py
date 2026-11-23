"""
Authentication and user management tests.
"""

import pytest


@pytest.mark.asyncio
async def test_register_user_success(client):
    payload = {
        "email": "ayush@example.com",
        "password": "Password123!",
        "full_name": "Ayush Sharma",
        "phone": "+919876543210"
    }
    response = await client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "ayush@example.com"
    assert data["user"]["full_name"] == "Ayush Sharma"
    assert data["user"]["currency"] == "INR"


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    payload = {
        "email": "duplicate@example.com",
        "password": "Password123!",
        "full_name": "First User",
    }
    res1 = await client.post("/api/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = await client.post("/api/auth/register", json=payload)
    assert res2.status_code == 409
    assert "already registered" in res2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_login_success(client):
    reg_payload = {
        "email": "login_test@example.com",
        "password": "StrongPassword123!",
        "full_name": "Login Tester",
    }
    await client.post("/api/auth/register", json=reg_payload)

    login_payload = {
        "email": "login_test@example.com",
        "password": "StrongPassword123!",
    }
    response = await client.post("/api/auth/login", json=login_payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "login_test@example.com"


@pytest.mark.asyncio
async def test_login_invalid_password(client):
    reg_payload = {
        "email": "wrong_pass@example.com",
        "password": "StrongPassword123!",
        "full_name": "User",
    }
    await client.post("/api/auth/register", json=reg_payload)

    response = await client.post("/api/auth/login", json={
        "email": "wrong_pass@example.com",
        "password": "WrongPassword999",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_profile(client):
    reg_payload = {
        "email": "profile@example.com",
        "password": "Password123!",
        "full_name": "Profile Owner",
    }
    reg_res = await client.post("/api/auth/register", json=reg_payload)
    token = reg_res.json()["access_token"]

    # Without token -> 401
    unauth_res = await client.get("/api/auth/me")
    assert unauth_res.status_code == 401

    # With valid token -> 200
    headers = {"Authorization": f"Bearer {token}"}
    me_res = await client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    profile = me_res.json()
    assert profile["email"] == "profile@example.com"
    assert profile["full_name"] == "Profile Owner"
