"""
Pytest configuration and async fixtures for LIFEOS backend tests.
"""

import asyncio
import os
import sys
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

# Ensure backend root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.models  # ensure all models are registered
from app.main import app as fastapi_app
from app.database import get_db
from app.database.session import Base
from app.models import Category

# In-memory SQLite async engine with StaticPool to share connection across async calls
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Create fresh tables for each test and provide a database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed default system categories
    async with TestingSessionLocal() as session:
        default_cats = [
            Category(name="Salary", icon="💰", color="#4CAF50", is_system=True, is_income=True),
            Category(name="Food & Dining", icon="🍽️", color="#FF6B6B", is_system=True, is_income=False),
            Category(name="Groceries", icon="🛒", color="#FF8E53", is_system=True, is_income=False),
            Category(name="Utilities", icon="💡", color="#A8E6CF", is_system=True, is_income=False),
            Category(name="Electricity", icon="⚡", color="#FDFFAB", is_system=True, is_income=False),
            Category(name="Rent", icon="🏠", color="#B4F8C8", is_system=True, is_income=False),
            Category(name="Shopping", icon="🛍️", color="#FEC89A", is_system=True, is_income=False),
            Category(name="Subscriptions", icon="📺", color="#CAB8FF", is_system=True, is_income=False),
            Category(name="Transportation", icon="🚗", color="#98D8C8", is_system=True, is_income=False),
            Category(name="Fuel", icon="⛽", color="#7FCDCD", is_system=True, is_income=False),
            Category(name="EMI & Loans", icon="🏦", color="#FFE66D", is_system=True, is_income=False),
            Category(name="Miscellaneous", icon="📌", color="#90A4AE", is_system=True, is_income=False),
        ]
        session.add_all(default_cats)
        await session.commit()

    async with TestingSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    """Provide an AsyncClient wired with the test database."""
    async def override_get_db():
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def test_user(db_session):
    """Provide a seeded test user."""
    from app.models import User
    user = User(
        email="test_user@example.com",
        hashed_password="hashed_pw_test_123",
        full_name="Test User",
        currency="INR",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture(scope="function")
async def auth_headers(test_user):
    """Provide authorization headers for test_user."""
    from app.services.auth_service import create_access_token
    token = create_access_token(user_id=test_user.id, email=test_user.email)
    return {"Authorization": f"Bearer {token}"}

