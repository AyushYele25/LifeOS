"""
LIFEOS Auth API Routes

Registration, login, and profile management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User
from app.schemas.schemas import UserCreate, UserLogin, UserResponse, Token
from app.services.auth_service import (
    hash_password, verify_password, create_access_token, get_current_user,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new LIFEOS user."""
    # Check if email exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        phone=user_data.phone,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    # Generate token
    token = create_access_token(user.id, user.email)

    return Token(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """Login and receive JWT token."""
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    token = create_access_token(user.id, user.email)

    return Token(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user's profile."""
    return UserResponse.model_validate(current_user)


@router.post("/demo-login", response_model=Token)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Instant 1-click login for evaluation demo user Priya Sharma with preloaded finances."""
    from datetime import date, timedelta
    from decimal import Decimal
    from app.models.models import (
        Account, AccountType, Transaction, TransactionType,
        Category, Loan, LoanType, LoanStatus, Insurance, InsuranceType,
        Subscription, SubscriptionStatus, SubscriptionFrequency,
        Event, EventType, EventStatus, Goal, GoalStatus,
    )
    from app.services.life_state_engine import LifeStateEngine

    demo_email = "demo.priya@lifeos.ai"
    result = await db.execute(select(User).where(User.email == demo_email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            email=demo_email,
            hashed_password=hash_password("DemoPassword123!"),
            full_name="Priya Sharma",
            phone="+91 9876543210",
            currency="INR",
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)

        # Create Primary Bank Account
        account = Account(
            user_id=user.id,
            name="HDFC Salary & Savings",
            account_type=AccountType.SAVINGS,
            current_balance=Decimal("380000.00"),
            is_primary=True,
        )
        db.add(account)
        await db.flush()
        await db.refresh(account)

        # Categories
        cats_res = await db.execute(select(Category))
        cat_map = {c.name: c.id for c in cats_res.scalars()}

        today = date.today()
        # Seed 3 months history
        for i in range(3):
            db.add(Transaction(
                user_id=user.id,
                account_id=account.id,
                category_id=cat_map.get("Salary"),
                date=today - timedelta(days=i * 30 + 5),
                description="Infosys Technologies Monthly Salary",
                amount=Decimal("120000.00"),
                transaction_type=TransactionType.CREDIT,
            ))
            db.add(Transaction(
                user_id=user.id,
                account_id=account.id,
                category_id=cat_map.get("Rent"),
                date=today - timedelta(days=i * 30 + 10),
                description="House Rent Transfer to Landlord",
                amount=Decimal("35000.00"),
                transaction_type=TransactionType.DEBIT,
            ))
            db.add(Transaction(
                user_id=user.id,
                account_id=account.id,
                category_id=cat_map.get("Groceries"),
                date=today - timedelta(days=i * 30 + 15),
                description="BigBasket Weekly Household Groceries",
                amount=Decimal("12000.00"),
                transaction_type=TransactionType.DEBIT,
            ))

        # Loan
        db.add(Loan(
            user_id=user.id,
            name="Car Loan HDFC",
            loan_type=LoanType.VEHICLE,
            lender="HDFC Bank",
            principal_amount=Decimal("600000.00"),
            outstanding_amount=Decimal("350000.00"),
            interest_rate=8.8,
            emi_amount=Decimal("14500.00"),
            start_date=today - timedelta(days=365),
            emi_day=10,
            status=LoanStatus.ACTIVE,
        ))

        # Insurance
        db.add(Insurance(
            user_id=user.id,
            name="Star Health Optima",
            insurance_type=InsuranceType.HEALTH,
            provider="Star Health",
            premium_amount=Decimal("18500.00"),
            sum_insured=Decimal("1000000.00"),
            expiry_date=today + timedelta(days=45),
        ))

        # Subscription
        db.add(Subscription(
            user_id=user.id,
            name="Netflix Premium",
            provider="Netflix",
            amount=Decimal("649.00"),
            frequency=SubscriptionFrequency.MONTHLY,
            next_billing_date=today + timedelta(days=12),
            status=SubscriptionStatus.ACTIVE,
        ))

        # Events
        db.add(Event(
            user_id=user.id,
            event_type=EventType.LOAN_EMI,
            title="Car Loan EMI Due",
            event_date=today + timedelta(days=10),
            amount=Decimal("14500.00"),
            status=EventStatus.UPCOMING,
            source_type="loan",
        ))

        # Goal
        db.add(Goal(
            user_id=user.id,
            name="Emergency Fund (6 Months)",
            description="Living expenses safety net reserve",
            target_amount=Decimal("300000.00"),
            current_amount=Decimal("180000.00"),
            status=GoalStatus.ACTIVE,
            priority=1,
        ))
        await db.commit()
        await db.refresh(user)

        # Initial Life State computation
        lse = LifeStateEngine(db, user.id)
        await lse.recompute_state()

    token = create_access_token(user.id, user.email)
    return Token(
        access_token=token,
        user=UserResponse.model_validate(user),
    )
