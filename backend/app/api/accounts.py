"""
LIFEOS Accounts API Routes

CRUD for bank accounts, credit cards, digital wallets, and cash reserves.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import UUID

from app.database import get_db
from app.models import User, Account
from app.schemas.schemas import AccountCreate, AccountResponse
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/accounts", tags=["Accounts"])


@router.get("", response_model=list[AccountResponse])
async def list_accounts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all active accounts for the current user."""
    result = await db.execute(
        select(Account).where(
            and_(Account.user_id == current_user.id, Account.is_active == True)
        ).order_by(Account.is_primary.desc(), Account.name)
    )
    return [AccountResponse.model_validate(a) for a in result.scalars()]


@router.post("", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
async def create_account(
    data: AccountCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new account (bank, card, wallet, cash)."""
    account = Account(user_id=current_user.id, **data.model_dump())
    db.add(account)
    await db.flush()
    await db.refresh(account)
    return AccountResponse.model_validate(account)


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    account_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Soft delete an account."""
    result = await db.execute(
        select(Account).where(
            and_(Account.id == account_id, Account.user_id == current_user.id)
        )
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    account.is_active = False
    await db.flush()
    return None
