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
        from datetime import datetime, timezone
        from app.models.models import LifeState, RiskLevel

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

        # Create Bank Accounts (Primary Savings + Emergency FD = ₹3,80,000 total)
        account = Account(
            user_id=user.id,
            name="HDFC Salary & Savings",
            account_type=AccountType.SAVINGS,
            bank_name="HDFC Bank",
            account_number_masked="4012",
            current_balance=Decimal("180000.00"),
            is_primary=True,
        )
        fd_account = Account(
            user_id=user.id,
            name="SBI Emergency Fixed Deposit",
            account_type=AccountType.SAVINGS,
            bank_name="State Bank of India",
            account_number_masked="8821",
            current_balance=Decimal("200000.00"),
            is_primary=False,
        )
        db.add_all([account, fd_account])
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
                category_ml_confidence=0.96,
            ))
            db.add(Transaction(
                user_id=user.id,
                account_id=account.id,
                category_id=cat_map.get("Rent"),
                date=today - timedelta(days=i * 30 + 10),
                description="House Rent Transfer to Landlord",
                amount=Decimal("35000.00"),
                transaction_type=TransactionType.DEBIT,
                category_ml_confidence=0.96,
            ))
            db.add(Transaction(
                user_id=user.id,
                account_id=account.id,
                category_id=cat_map.get("Groceries"),
                date=today - timedelta(days=i * 30 + 15),
                description="BigBasket Weekly Household Groceries",
                amount=Decimal("12000.00"),
                transaction_type=TransactionType.DEBIT,
                category_ml_confidence=0.96,
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
            total_emis=36,
            emis_paid=12,
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

        # Active Subscriptions (for Zombie Hunter & Life State)
        db.add_all([
            Subscription(
                user_id=user.id,
                name="Netflix Premium",
                provider="Netflix",
                amount=Decimal("649.00"),
                frequency=SubscriptionFrequency.MONTHLY,
                next_billing_date=today + timedelta(days=12),
                category="entertainment",
                status=SubscriptionStatus.ACTIVE,
            ),
            Subscription(
                user_id=user.id,
                name="Cult.fit Pro Gym",
                provider="Cult.fit",
                amount=Decimal("1499.00"),
                frequency=SubscriptionFrequency.MONTHLY,
                next_billing_date=today + timedelta(days=18),
                category="fitness",
                status=SubscriptionStatus.ACTIVE,
            ),
            Subscription(
                user_id=user.id,
                name="Amazon Prime Membership",
                provider="Amazon",
                amount=Decimal("1499.00"),
                frequency=SubscriptionFrequency.YEARLY,
                next_billing_date=today + timedelta(days=65),
                category="shopping",
                status=SubscriptionStatus.ACTIVE,
            ),
        ])

        # Events (within 7 days + 10 days + 45 days for Radar & Horizons)
        db.add_all([
            Event(
                user_id=user.id,
                event_type=EventType.BILL_DUE,
                title="BESCOM Electricity Bill Due",
                event_date=today + timedelta(days=4),
                amount=Decimal("2150.00"),
                status=EventStatus.UPCOMING,
                source_type="bill",
            ),
            Event(
                user_id=user.id,
                event_type=EventType.LOAN_EMI,
                title="Car Loan EMI Due",
                event_date=today + timedelta(days=10),
                amount=Decimal("14500.00"),
                status=EventStatus.UPCOMING,
                source_type="loan",
            ),
            Event(
                user_id=user.id,
                event_type=EventType.INSURANCE_RENEWAL,
                title="Star Health Optima Renewal",
                event_date=today + timedelta(days=45),
                amount=Decimal("18500.00"),
                status=EventStatus.UPCOMING,
                source_type="insurance",
            ),
        ])

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

        # Historical Baseline Snapshot (30 days ago) for "What Changed?" comparison
        db.add(LifeState(
            user_id=user.id,
            snapshot_date=datetime.now(timezone.utc) - timedelta(days=30),
            total_balance=Decimal("325000.00"),
            monthly_income=Decimal("120000.00"),
            monthly_expenses=Decimal("52500.00"),
            monthly_savings=Decimal("67500.00"),
            savings_rate=0.5625,
            emergency_runway_months=6.2,
            financial_risk_score=24.0,
            risk_level=RiskLevel.LOW,
        ))

        await db.commit()
        await db.refresh(user)

        # Current Life State computation (automatically calculates delta vs 30d ago)
        lse = LifeStateEngine(db, user.id)
        await lse.recompute_state()

    token = create_access_token(user.id, user.email)
    return Token(
        access_token=token,
        user=UserResponse.model_validate(user),
    )
