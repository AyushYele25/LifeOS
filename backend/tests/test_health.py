"""
Health and basic configuration tests.
"""

import pytest
from sqlalchemy import select, func
from app.models import Category


@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LIFEOS"
    assert data["status"] == "operational"


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_default_categories_seeded(db_session):
    result = await db_session.execute(
        select(func.count(Category.id)).where(Category.is_system == True)
    )
    count = result.scalar()
    assert count >= 4
