"""
LIFEOS SQLAlchemy Models

Complete database schema for the Life Operating System.
Every table is designed to feed into the Life State Engine.
"""

import uuid
from datetime import datetime, date, timezone

def utc_now():
    return datetime.now(timezone.utc)
from decimal import Decimal
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Date,
    Text, ForeignKey, Enum, Index, JSON, Numeric, UniqueConstraint, Uuid,
)
UUID = Uuid
JSONB = JSON
from sqlalchemy.orm import relationship
from app.database.session import Base
import enum


# ═══════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════

class TransactionType(str, enum.Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class AccountType(str, enum.Enum):
    SAVINGS = "savings"
    CURRENT = "current"
    CREDIT_CARD = "credit_card"
    WALLET = "wallet"
    CASH = "cash"


class AssetType(str, enum.Enum):
    VEHICLE = "vehicle"
    ELECTRONICS = "electronics"
    APPLIANCE = "appliance"
    FURNITURE = "furniture"
    JEWELRY = "jewelry"
    PROPERTY = "property"
    OTHER = "other"


class LoanType(str, enum.Enum):
    HOME = "home"
    VEHICLE = "vehicle"
    PERSONAL = "personal"
    EDUCATION = "education"
    CREDIT_CARD = "credit_card"
    OTHER = "other"


class LoanStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    DEFAULTED = "defaulted"


class InsuranceType(str, enum.Enum):
    VEHICLE = "vehicle"
    HEALTH = "health"
    LIFE = "life"
    HOME = "home"
    TRAVEL = "travel"
    OTHER = "other"


class SubscriptionFrequency(str, enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    HALF_YEARLY = "half_yearly"
    YEARLY = "yearly"


class SubscriptionStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class EventType(str, enum.Enum):
    INSURANCE_RENEWAL = "insurance_renewal"
    LOAN_EMI = "loan_emi"
    SUBSCRIPTION_PAYMENT = "subscription_payment"
    BILL_DUE = "bill_due"
    WARRANTY_EXPIRY = "warranty_expiry"
    VEHICLE_SERVICE = "vehicle_service"
    SALARY_CREDIT = "salary_credit"
    INVESTMENT_MATURITY = "investment_maturity"
    TAX_DEADLINE = "tax_deadline"
    CUSTOM = "custom"


class EventStatus(str, enum.Enum):
    UPCOMING = "upcoming"
    DUE = "due"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, enum.Enum):
    ANOMALY = "anomaly"
    RENEWAL = "renewal"
    BUDGET_EXCEEDED = "budget_exceeded"
    GOAL_PROGRESS = "goal_progress"
    RISK_CHANGE = "risk_change"
    FORECAST_WARNING = "forecast_warning"
    DOCUMENT_EXPIRY = "document_expiry"
    STATE_CHANGE = "state_change"


class DocumentType(str, enum.Enum):
    INSURANCE = "insurance"
    LOAN = "loan"
    WARRANTY = "warranty"
    BILL = "bill"
    RECEIPT = "receipt"
    SALARY_SLIP = "salary_slip"
    BANK_STATEMENT = "bank_statement"
    ID_DOCUMENT = "id_document"
    OTHER = "other"


class GoalStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    PAUSED = "paused"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=True)
    currency = Column(String(10), default="INR")
    timezone = Column(String(50), default="Asia/Kolkata")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="user", cascade="all, delete-orphan")
    loans = relationship("Loan", back_populates="user", cascade="all, delete-orphan")
    insurance_policies = relationship("Insurance", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="user", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    life_states = relationship("LifeState", back_populates="user", cascade="all, delete-orphan")
    categories = relationship("Category", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    simulations = relationship("Simulation", back_populates="user", cascade="all, delete-orphan")
    recurring_patterns = relationship("RecurringPattern", back_populates="user", cascade="all, delete-orphan")


class Account(Base):
    """Bank accounts, wallets, credit cards."""
    __tablename__ = "accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    account_type = Column(Enum(AccountType), nullable=False)
    bank_name = Column(String(255), nullable=True)
    account_number_masked = Column(String(20), nullable=True)  # Last 4 digits only
    current_balance = Column(Numeric(14, 2), default=0)
    is_primary = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")


class Category(Base):
    """Transaction categories — system defaults + user custom."""
    __tablename__ = "categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # NULL = system category
    name = Column(String(100), nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("categories.id"), nullable=True)
    icon = Column(String(50), nullable=True)
    color = Column(String(7), nullable=True)  # Hex color
    is_system = Column(Boolean, default=False)
    is_income = Column(Boolean, default=False)  # True for income categories
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="categories")
    parent = relationship("Category", remote_side=[id])
    transactions = relationship("Transaction", back_populates="category")


class Transaction(Base):
    """Every financial transaction — the primary data source."""
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=True)
    category_id = Column(UUID(as_uuid=True), ForeignKey("categories.id"), nullable=True)

    date = Column(Date, nullable=False, index=True)
    description = Column(String(500), nullable=False)
    amount = Column(Numeric(14, 2), nullable=False)
    transaction_type = Column(Enum(TransactionType), nullable=False)

    # Classification
    merchant = Column(String(255), nullable=True)
    category_ml_confidence = Column(Float, nullable=True)  # ML classification confidence
    is_classified_by_user = Column(Boolean, default=False)  # Human-in-the-loop feedback

    # Recurring detection
    is_recurring = Column(Boolean, default=False)
    recurring_group_id = Column(UUID(as_uuid=True), nullable=True)

    # Anomaly detection
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)

    # Source tracking
    source = Column(String(50), default="manual")  # csv, manual, api, upi
    raw_data = Column(JSONB, nullable=True)  # Original row from CSV/API
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Indexes for performance
    __table_args__ = (
        Index("ix_transactions_user_date", "user_id", "date"),
        Index("ix_transactions_user_category", "user_id", "category_id"),
        Index("ix_transactions_user_merchant", "user_id", "merchant"),
        Index("ix_transactions_recurring", "user_id", "is_recurring"),
    )

    user = relationship("User", back_populates="transactions")
    account = relationship("Account", back_populates="transactions")
    category = relationship("Category", back_populates="transactions")


class Asset(Base):
    """Physical and digital assets owned by the user."""
    __tablename__ = "assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    brand = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    serial_number = Column(String(255), nullable=True)

    purchase_date = Column(Date, nullable=True)
    purchase_price = Column(Numeric(14, 2), nullable=True)
    current_value = Column(Numeric(14, 2), nullable=True)

    warranty_expiry = Column(Date, nullable=True)
    insurance_expiry = Column(Date, nullable=True)
    next_service_date = Column(Date, nullable=True)
    service_interval_days = Column(Integer, nullable=True)

    # Total Cost of Ownership tracking
    total_maintenance_cost = Column(Numeric(14, 2), default=0)
    total_fuel_cost = Column(Numeric(14, 2), default=0)  # For vehicles
    total_insurance_cost = Column(Numeric(14, 2), default=0)

    metadata_json = Column(JSONB, nullable=True)  # Flexible extra fields
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="assets")


class Loan(Base):
    """Active and closed loans."""
    __tablename__ = "loans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    loan_type = Column(Enum(LoanType), nullable=False)
    lender = Column(String(255), nullable=True)
    account_number_masked = Column(String(20), nullable=True)

    principal_amount = Column(Numeric(14, 2), nullable=False)
    outstanding_amount = Column(Numeric(14, 2), nullable=True)
    interest_rate = Column(Float, nullable=True)  # Annual percentage
    emi_amount = Column(Numeric(14, 2), nullable=True)

    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    emi_day = Column(Integer, nullable=True)  # Day of month for EMI
    total_emis = Column(Integer, nullable=True)
    emis_paid = Column(Integer, default=0)

    status = Column(Enum(LoanStatus), default=LoanStatus.ACTIVE)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="loans")


class Insurance(Base):
    """Insurance policies."""
    __tablename__ = "insurance_policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    insurance_type = Column(Enum(InsuranceType), nullable=False)
    provider = Column(String(255), nullable=True)
    policy_number = Column(String(100), nullable=True)

    premium_amount = Column(Numeric(14, 2), nullable=True)
    premium_frequency = Column(Enum(SubscriptionFrequency), default=SubscriptionFrequency.YEARLY)
    sum_insured = Column(Numeric(14, 2), nullable=True)

    start_date = Column(Date, nullable=True)
    expiry_date = Column(Date, nullable=True)
    next_premium_date = Column(Date, nullable=True)

    # Link to asset (e.g., vehicle insurance)
    asset_id = Column(UUID(as_uuid=True), ForeignKey("assets.id"), nullable=True)

    is_active = Column(Boolean, default=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="insurance_policies")


class Subscription(Base):
    """Recurring subscriptions and services."""
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    provider = Column(String(255), nullable=True)
    amount = Column(Numeric(14, 2), nullable=False)
    frequency = Column(Enum(SubscriptionFrequency), nullable=False)

    start_date = Column(Date, nullable=True)
    next_billing_date = Column(Date, nullable=True)
    last_billing_date = Column(Date, nullable=True)

    category = Column(String(100), nullable=True)  # entertainment, utility, software, etc.
    status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE)

    # Auto-detected from transaction patterns
    is_auto_detected = Column(Boolean, default=False)
    detection_confidence = Column(Float, nullable=True)

    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="subscriptions")


class Event(Base):
    """
    Future obligations and events.
    This is one of LIFEOS's most important tables.
    Every document, loan, insurance, subscription generates events.
    """
    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    event_type = Column(Enum(EventType), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    event_date = Column(Date, nullable=False, index=True)
    amount = Column(Numeric(14, 2), nullable=True)
    status = Column(Enum(EventStatus), default=EventStatus.UPCOMING)
    confidence = Column(Float, default=1.0)  # How confident we are this will happen

    # Source tracking — what generated this event
    source_type = Column(String(50), nullable=True)  # loan, insurance, subscription, document, prediction
    source_id = Column(UUID(as_uuid=True), nullable=True)

    # Recurrence
    is_recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(JSONB, nullable=True)  # {frequency, interval, end_date}

    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    __table_args__ = (
        Index("ix_events_user_date", "user_id", "event_date"),
        Index("ix_events_user_status", "user_id", "status"),
    )

    user = relationship("User", back_populates="events")


class Goal(Base):
    """Financial goals — emergency fund, savings targets, etc."""
    __tablename__ = "goals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    target_amount = Column(Numeric(14, 2), nullable=False)
    current_amount = Column(Numeric(14, 2), default=0)
    target_date = Column(Date, nullable=True)
    status = Column(Enum(GoalStatus), default=GoalStatus.ACTIVE)
    priority = Column(Integer, default=1)  # 1 = highest
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="goals")


class Document(Base):
    """Uploaded documents with extracted intelligence."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)

    document_type = Column(Enum(DocumentType), nullable=True)
    classification_confidence = Column(Float, nullable=True)

    # Extracted information
    extracted_text = Column(Text, nullable=True)
    extracted_entities = Column(JSONB, nullable=True)  # Structured extraction result
    summary = Column(Text, nullable=True)

    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)

    # ChromaDB reference
    chroma_collection = Column(String(255), nullable=True)
    chroma_ids = Column(JSONB, nullable=True)  # List of chunk IDs in ChromaDB

    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="documents")


class Alert(Base):
    """System-generated alerts and notifications."""
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)

    # What triggered this alert
    source_type = Column(String(50), nullable=True)
    source_id = Column(UUID(as_uuid=True), nullable=True)

    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    action_url = Column(String(500), nullable=True)

    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="alerts")


class LifeState(Base):
    """
    Snapshot of the user's complete financial state at a point in time.
    This is the heart of LIFEOS — the Life State Engine.

    A new snapshot is created whenever significant state changes occur.
    This enables "What changed?" comparisons.
    """
    __tablename__ = "life_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    snapshot_date = Column(DateTime, default=utc_now, index=True)

    # ─── Money ────────────────────────────────────────
    total_balance = Column(Numeric(14, 2), default=0)
    monthly_income = Column(Numeric(14, 2), default=0)
    monthly_expenses = Column(Numeric(14, 2), default=0)
    monthly_savings = Column(Numeric(14, 2), default=0)
    savings_rate = Column(Float, default=0)  # savings / income

    # ─── Obligations ──────────────────────────────────
    upcoming_obligations_30d = Column(Numeric(14, 2), default=0)
    upcoming_obligations_60d = Column(Numeric(14, 2), default=0)
    upcoming_obligations_90d = Column(Numeric(14, 2), default=0)
    active_loans_count = Column(Integer, default=0)
    total_emi_burden = Column(Numeric(14, 2), default=0)
    active_subscriptions_count = Column(Integer, default=0)
    total_subscription_burden = Column(Numeric(14, 2), default=0)

    # ─── Assets ───────────────────────────────────────
    total_assets_count = Column(Integer, default=0)
    total_asset_value = Column(Numeric(14, 2), default=0)

    # ─── Risk ─────────────────────────────────────────
    financial_risk_score = Column(Float, default=0)  # 0-100
    risk_level = Column(Enum(RiskLevel), default=RiskLevel.LOW)
    emergency_runway_months = Column(Float, default=0)  # How many months of expenses covered

    # ─── Risk Breakdown ───────────────────────────────
    risk_breakdown = Column(JSONB, nullable=True)
    # Example: {
    #   "emi_income_ratio": {"score": 12, "reason": "High EMI/income ratio"},
    #   "emergency_fund": {"score": -10, "reason": "Stable emergency fund"},
    #   ...
    # }

    # ─── Category Breakdown ──────────────────────────
    expense_by_category = Column(JSONB, nullable=True)  # {category: amount}
    income_by_source = Column(JSONB, nullable=True)

    # ─── Changes from previous state ─────────────────
    changes_from_previous = Column(JSONB, nullable=True)
    # Example: [
    #   {"field": "monthly_expenses", "old": 38000, "new": 43800, "change_pct": 15.3},
    #   {"type": "new_recurring", "description": "Netflix ₹649/month detected"},
    #   ...
    # ]

    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    __table_args__ = (
        Index("ix_life_states_user_date", "user_id", "snapshot_date"),
    )

    user = relationship("User", back_populates="life_states")


class Prediction(Base):
    """ML-generated predictions — expenses, cashflow, etc."""
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    prediction_type = Column(String(50), nullable=False)  # expense_forecast, cashflow, savings
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)

    predicted_amount = Column(Numeric(14, 2), nullable=False)
    confidence_lower = Column(Numeric(14, 2), nullable=True)  # Lower bound
    confidence_upper = Column(Numeric(14, 2), nullable=True)  # Upper bound
    confidence_level = Column(Float, default=0.95)

    model_used = Column(String(100), nullable=True)
    features_used = Column(JSONB, nullable=True)
    actual_amount = Column(Numeric(14, 2), nullable=True)  # Filled in after period ends

    breakdown = Column(JSONB, nullable=True)  # By category breakdown
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_user_type", "user_id", "prediction_type"),
    )


class Simulation(Base):
    """Decision engine simulations — 'Can I afford X?' scenarios."""
    __tablename__ = "simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(500), nullable=False)
    simulation_type = Column(String(50), nullable=False)  # purchase, loan, stress_test

    # Input parameters
    input_params = Column(JSONB, nullable=False)
    # Example for purchase: {"item": "Laptop", "amount": 70000, "emi_months": 12}

    # Scenarios
    scenarios = Column(JSONB, nullable=False)
    # Example: [
    #   {"name": "Buy Now", "impact": {...}, "risk_after": "MEDIUM"},
    #   {"name": "Wait 3 Months", "impact": {...}, "risk_after": "LOW"},
    #   {"name": "EMI 12 months", "impact": {...}, "risk_after": "MEDIUM"},
    # ]

    recommendation = Column(Text, nullable=True)
    life_state_at_time = Column(UUID(as_uuid=True), nullable=True)  # Reference to LifeState

    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="simulations")


class RecurringPattern(Base):
    """Detected recurring payment patterns."""
    __tablename__ = "recurring_patterns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    merchant = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)
    average_amount = Column(Numeric(14, 2), nullable=False)
    frequency = Column(Enum(SubscriptionFrequency), nullable=False)
    average_interval_days = Column(Float, nullable=True)

    last_transaction_date = Column(Date, nullable=True)
    next_expected_date = Column(Date, nullable=True)
    confidence = Column(Float, nullable=False)

    transaction_count = Column(Integer, default=0)
    is_confirmed = Column(Boolean, default=False)  # User confirmed

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="recurring_patterns")

    __table_args__ = (
        Index("ix_recurring_user_merchant", "user_id", "merchant"),
    )


class AuditLog(Base):
    """Audit log for user actions, data imports, and system state snapshots."""
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(String(64), nullable=True)
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

    # Relationship
    user = relationship("User", back_populates="audit_logs")

