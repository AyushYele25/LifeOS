"""
LIFEOS Loans API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import UUID

from app.database import get_db
from app.models import User, Loan
from app.models.models import LoanStatus
from app.schemas.schemas import LoanCreate, LoanResponse
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/loans", tags=["Loans"])


@router.get("", response_model=list[LoanResponse])
async def list_loans(
    include_closed: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Loan).where(Loan.user_id == current_user.id)
    if not include_closed:
        query = query.where(Loan.status == LoanStatus.ACTIVE)
    result = await db.execute(query)
    return [LoanResponse.model_validate(l) for l in result.scalars()]


@router.post("", response_model=LoanResponse, status_code=status.HTTP_201_CREATED)
async def create_loan(
    data: LoanCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    loan = Loan(user_id=current_user.id, **data.model_dump())
    db.add(loan)
    await db.flush()
    await db.refresh(loan)
    return LoanResponse.model_validate(loan)


@router.get("/{loan_id}", response_model=LoanResponse)
async def get_loan(
    loan_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Loan).where(and_(Loan.id == loan_id, Loan.user_id == current_user.id))
    )
    loan = result.scalar_one_or_none()
    if not loan:
        raise HTTPException(status_code=404, detail="Loan not found")
    return LoanResponse.model_validate(loan)
