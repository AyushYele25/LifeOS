"""
LIFEOS Assets API Routes

CRUD for physical/digital assets with TCO tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import UUID

from app.database import get_db
from app.models import User, Asset
from app.schemas.schemas import AssetCreate, AssetResponse
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/assets", tags=["Assets"])


@router.get("", response_model=list[AssetResponse])
async def list_assets(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all assets."""
    result = await db.execute(
        select(Asset).where(
            and_(Asset.user_id == current_user.id, Asset.is_active == True)
        )
    )
    return [AssetResponse.model_validate(a) for a in result.scalars()]


@router.post("", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
async def create_asset(
    data: AssetCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new asset."""
    asset = Asset(user_id=current_user.id, **data.model_dump())
    db.add(asset)
    await db.flush()
    await db.refresh(asset)
    return AssetResponse.model_validate(asset)


@router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(
    asset_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Asset).where(
            and_(Asset.id == asset_id, Asset.user_id == current_user.id)
        )
    )
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return AssetResponse.model_validate(asset)


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(
    asset_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Asset).where(
            and_(Asset.id == asset_id, Asset.user_id == current_user.id)
        )
    )
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    asset.is_active = False
    await db.flush()
