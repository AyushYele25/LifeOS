"""
LIFEOS Pydantic Schemas

Request/Response validation with strict typing.
The LLM never writes directly to the database — everything goes through these schemas.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID
from app.models.models import (
    TransactionType, AccountType, AssetType, LoanType, LoanStatus,
    InsuranceType, SubscriptionFrequency, SubscriptionStatus,
    EventType, EventStatus, AlertSeverity, AlertType,
    DocumentType, GoalStatus, RiskLevel,
)


# ═══════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=255)
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    phone: Optional[str]
    currency: str
    timezone: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ═══════════════════════════════════════════════════════
# ACCOUNT
# ═══════════════════════════════════════════════════════

class AccountCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    account_type: AccountType
    bank_name: Optional[str] = None
    account_number_masked: Optional[str] = None
    current_balance: Decimal = Decimal("0")
    is_primary: bool = False


class AccountResponse(BaseModel):
    id: UUID
    name: str
    account_type: AccountType
    bank_name: Optional[str]
    account_number_masked: Optional[str]
    current_balance: Decimal
    is_primary: bool
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════
# TRANSACTION
# ═══════════════════════════════════════════════════════

class TransactionCreate(BaseModel):
    date: date
    description: str = Field(..., min_length=1, max_length=500)
    amount: Decimal = Field(..., gt=0)
    transaction_type: TransactionType
    account_id: Optional[UUID] = None
    category_id: Optional[UUID] = None
    merchant: Optional[str] = None
    notes: Optional[str] = None
    source: str = "manual"


class TransactionUpdate(BaseModel):
    category_id: Optional[UUID] = None
    merchant: Optional[str] = None
    notes: Optional[str] = None
    is_recurring: Optional[bool] = None
    is_anomaly: Optional[bool] = None


class CategoryCorrectionRequest(BaseModel):
    category_id: UUID
    notes: Optional[str] = None


class TransactionResponse(BaseModel):
    id: UUID
    date: date
    description: str
    amount: Decimal
    transaction_type: TransactionType
    merchant: Optional[str]
    category_id: Optional[UUID]
    category_ml_confidence: Optional[float]
    is_classified_by_user: bool
    is_recurring: bool
    is_anomaly: bool
    anomaly_score: Optional[float]
    source: str
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class TransactionBulkImport(BaseModel):
    """For CSV import — batch of transactions."""
    account_id: Optional[UUID] = None
    transactions: List[TransactionCreate]


class TransactionFilter(BaseModel):
    """Query filters for transaction listing."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    transaction_type: Optional[TransactionType] = None
    category_id: Optional[UUID] = None
    merchant: Optional[str] = None
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None
    is_recurring: Optional[bool] = None
    is_anomaly: Optional[bool] = None
    limit: int = Field(default=50, le=500)
    offset: int = Field(default=0, ge=0)


# ═══════════════════════════════════════════════════════
# ASSET
# ═══════════════════════════════════════════════════════

class AssetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    asset_type: AssetType
    brand: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    purchase_date: Optional[date] = None
    purchase_price: Optional[Decimal] = None
    current_value: Optional[Decimal] = None
    warranty_expiry: Optional[date] = None
    insurance_expiry: Optional[date] = None
    next_service_date: Optional[date] = None
    service_interval_days: Optional[int] = None


class AssetResponse(BaseModel):
    id: UUID
    name: str
    asset_type: AssetType
    brand: Optional[str]
    model: Optional[str]
    purchase_date: Optional[date]
    purchase_price: Optional[Decimal]
    current_value: Optional[Decimal]
    warranty_expiry: Optional[date]
    insurance_expiry: Optional[date]
    next_service_date: Optional[date]
    service_interval_days: Optional[int]
    total_maintenance_cost: Decimal
    total_fuel_cost: Decimal
    total_insurance_cost: Decimal
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════
# LOAN
# ═══════════════════════════════════════════════════════

class LoanCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    loan_type: LoanType
    lender: Optional[str] = None
    principal_amount: Decimal = Field(..., gt=0)
    outstanding_amount: Optional[Decimal] = None
    interest_rate: Optional[float] = None
    emi_amount: Optional[Decimal] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    emi_day: Optional[int] = Field(None, ge=1, le=31)
    total_emis: Optional[int] = None
    emis_paid: int = 0


class LoanResponse(BaseModel):
    id: UUID
    name: str
    loan_type: LoanType
    lender: Optional[str]
    principal_amount: Decimal
    outstanding_amount: Optional[Decimal]
    interest_rate: Optional[float]
    emi_amount: Optional[Decimal]
    start_date: Optional[date]
    end_date: Optional[date]
    emi_day: Optional[int]
    total_emis: Optional[int]
    emis_paid: int
    status: LoanStatus
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════
# SUBSCRIPTION
# ═══════════════════════════════════════════════════════

class SubscriptionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    provider: Optional[str] = None
    amount: Decimal = Field(..., gt=0)
    frequency: SubscriptionFrequency
    start_date: Optional[date] = None
    next_billing_date: Optional[date] = None
    category: Optional[str] = None


class SubscriptionResponse(BaseModel):
    id: UUID
    name: str
    provider: Optional[str]
    amount: Decimal
    frequency: SubscriptionFrequency
    start_date: Optional[date]
    next_billing_date: Optional[date]
    last_billing_date: Optional[date]
    category: Optional[str]
    status: SubscriptionStatus
    is_auto_detected: bool
    detection_confidence: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════
# EVENT
# ═══════════════════════════════════════════════════════

class EventCreate(BaseModel):
    event_type: EventType
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    event_date: date
    amount: Optional[Decimal] = None
    is_recurring: bool = False
    recurrence_pattern: Optional[Dict[str, Any]] = None


class EventResponse(BaseModel):
    id: UUID
    event_type: EventType
    title: str
    description: Optional[str] = None
    event_date: date
    amount: Optional[Decimal] = None
    status: EventStatus
    confidence: float
    source_type: Optional[str] = None
    source_id: Optional[UUID] = None
    is_recurring: bool = False
    recurrence_pattern: Optional[Dict[str, Any]] = None
    days_until: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EventStatusUpdate(BaseModel):
    status: EventStatus


class HorizonTotals(BaseModel):
    total_30d: Decimal
    total_60d: Decimal
    total_90d: Decimal


class HorizonSummaryResponse(BaseModel):
    as_of_date: date
    totals: HorizonTotals
    by_type: Dict[str, Decimal]
    by_month: Dict[str, List[EventResponse]]
    upcoming_events_count: int
    overdue_events_count: int
    timeline: List[EventResponse]


class EventSyncResult(BaseModel):
    scanned_sources: Dict[str, int]
    new_events_created: int
    overdue_events_marked: int
    events: List[EventResponse]


# ═══════════════════════════════════════════════════════
# GOAL
# ═══════════════════════════════════════════════════════

class GoalCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    target_amount: Decimal = Field(..., gt=0)
    current_amount: Decimal = Decimal("0")
    target_date: Optional[date] = None
    priority: int = Field(default=1, ge=1, le=10)


class GoalUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    target_amount: Optional[Decimal] = Field(None, gt=0)
    current_amount: Optional[Decimal] = None
    target_date: Optional[date] = None
    status: Optional[GoalStatus] = None
    priority: Optional[int] = Field(None, ge=1, le=10)


class GoalResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    target_amount: Decimal
    current_amount: Decimal
    target_date: Optional[date]
    status: GoalStatus
    priority: int
    created_at: datetime

    model_config = {"from_attributes": True}

    @property
    def progress_pct(self) -> float:
        if self.target_amount > 0:
            return float(self.current_amount / self.target_amount * 100)
        return 0.0


# ═══════════════════════════════════════════════════════
# ALERT
# ═══════════════════════════════════════════════════════

class AlertResponse(BaseModel):
    id: UUID
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    is_read: bool
    is_dismissed: bool
    source_type: Optional[str] = None
    source_id: Optional[UUID] = None
    action_url: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertDismissRequest(BaseModel):
    mark_as_expected: bool = False
    feedback: Optional[str] = None


class AlertSummaryResponse(BaseModel):
    total_active: int
    critical_count: int
    warning_count: int
    info_count: int
    recent_alerts: List[AlertResponse]


class AnomalyDetectionRunResponse(BaseModel):
    scanned_transactions: int
    anomalies_detected: int
    new_alerts_created: int
    alerts: List[AlertResponse]


# ═══════════════════════════════════════════════════════
# LIFE STATE
# ═══════════════════════════════════════════════════════

class LifeStateResponse(BaseModel):
    id: UUID
    snapshot_date: datetime
    total_balance: Decimal
    monthly_income: Decimal
    monthly_expenses: Decimal
    monthly_savings: Decimal
    savings_rate: float
    upcoming_obligations_30d: Decimal
    upcoming_obligations_60d: Decimal
    upcoming_obligations_90d: Decimal
    active_loans_count: int
    total_emi_burden: Decimal
    active_subscriptions_count: int
    total_subscription_burden: Decimal
    total_assets_count: int
    total_asset_value: Decimal
    financial_risk_score: float
    risk_level: RiskLevel
    emergency_runway_months: float
    risk_breakdown: Optional[Dict[str, Any]]
    expense_by_category: Optional[Dict[str, Any]]
    income_by_source: Optional[Dict[str, Any]]
    changes_from_previous: Optional[List[Dict[str, Any]]]

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════
# FORECAST
# ═══════════════════════════════════════════════════════

class CategoryForecast(BaseModel):
    category: str
    contractual_floor: Decimal
    p10_lower: Decimal
    p50_expected: Decimal
    p90_upper: Decimal
    seasonal_factor: float
    confidence: float


class HorizonPeriodForecast(BaseModel):
    days: int
    period_end: date
    total_floor: Decimal
    p10_lower: Decimal
    p50_expected: Decimal
    p90_upper: Decimal
    categories: List[CategoryForecast]


class ForecastResponse(BaseModel):
    as_of_date: date
    horizon_30d: HorizonPeriodForecast
    horizon_60d: HorizonPeriodForecast
    horizon_90d: HorizonPeriodForecast
    methodology: str
    active_seasonal_factors: List[Dict[str, Any]]


class ForecastBacktestResponse(BaseModel):
    evaluated_period: str
    actual_spend: Decimal
    forecasted_spend: Decimal
    mape: float
    accuracy_pct: float
    category_accuracies: Dict[str, float]


# ═══════════════════════════════════════════════════════
# FINANCIAL RISK & STRESS TESTING
# ═══════════════════════════════════════════════════════

class RiskBreakdownItem(BaseModel):
    factor: str
    score: float
    weight: float
    status: str  # "OK", "WARNING", "CRITICAL"
    message: str


class StressTestScenarioResult(BaseModel):
    scenario_name: str
    description: str
    monthly_cashflow_delta: Decimal
    liquid_balance_after: Decimal
    runway_months_after: float
    risk_level_after: RiskLevel
    is_survivable_6_months: bool
    details: Dict[str, Any]


class CustomStressTestRequest(BaseModel):
    income_drop_pct: Optional[float] = Field(default=0.0, ge=0.0, le=100.0)
    emergency_expense: Optional[Decimal] = Field(default=Decimal("0.00"), ge=0.0)
    interest_rate_hike_bps: Optional[int] = Field(default=0, ge=0, le=1000)
    inflation_pct: Optional[float] = Field(default=0.0, ge=0.0, le=100.0)
    duration_months: Optional[int] = Field(default=3, ge=1, le=24)


class FinancialRiskResponse(BaseModel):
    composite_risk_score: float
    risk_level: RiskLevel
    risk_grade: str
    dti_ratio: float
    savings_rate: float
    emergency_runway_months: float
    vulnerabilities: List[str]
    recommendations: List[str]
    breakdown: List[RiskBreakdownItem]
    stress_tests: List[StressTestScenarioResult]


# ═══════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════

class SimulationRequest(BaseModel):
    """Can I afford X? / What happens if Y?"""
    simulation_type: str = Field(..., pattern="^(purchase|loan|stress_test|salary_change|new_expense)$")
    title: str
    params: Dict[str, Any]
    # Examples:
    # purchase: {"item": "Laptop", "amount": 70000, "emi_months": 12}
    # stress_test: {"scenario": "salary_loss", "months": 3}
    # salary_change: {"new_salary": 55000}
    # new_expense: {"description": "Gym", "amount": 2500, "frequency": "monthly"}


class SimulationScenario(BaseModel):
    name: str
    description: str
    savings_after: Decimal
    monthly_cashflow_impact: Decimal
    risk_level_after: RiskLevel
    emergency_runway_months: float
    details: Dict[str, Any]


class SimulationResponse(BaseModel):
    id: UUID
    title: str
    simulation_type: str
    input_params: Dict[str, Any]
    scenarios: List[SimulationScenario]
    recommendation: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════
# DOCUMENT
# ═══════════════════════════════════════════════════════

class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    original_filename: str
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    document_type: Optional[DocumentType] = None
    classification_confidence: Optional[float] = None
    extracted_text: Optional[str] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    is_processed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUpdate(BaseModel):
    document_type: Optional[DocumentType] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


class DocumentProcessResult(BaseModel):
    document: DocumentResponse
    synced_records: Dict[str, Any]
    generated_events: List[Dict[str, Any]]


# ═══════════════════════════════════════════════════════
# DOCUMENT RAG
# ═══════════════════════════════════════════════════════

class RAGCitation(BaseModel):
    document_id: UUID
    filename: str
    document_type: Optional[str] = None
    page_number: Optional[int] = 1
    chunk_index: int
    text_snippet: str
    relevance_score: float


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Semantic question about uploaded user documents")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of relevant chunks to retrieve")
    document_type: Optional[DocumentType] = Field(None, description="Filter by document type")
    document_id: Optional[UUID] = Field(None, description="Filter to a specific document")


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[RAGCitation]
    has_conflicts: bool = False
    conflict_details: Optional[str] = None
    model_used: str = "chroma-local-embedding"
    retrieved_chunk_count: int = 0


class RAGIndexResult(BaseModel):
    document_id: UUID
    filename: str
    chunks_indexed: int
    collection_name: str
    status: str = "success"


class RAGReindexResult(BaseModel):
    documents_processed: int
    total_chunks_indexed: int
    status: str = "success"


# ═══════════════════════════════════════════════════════
# AGENTIC AI
# ═══════════════════════════════════════════════════════

class AgentChatRequest(BaseModel):
    """Send a natural language message to the LIFEOS agentic assistant."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's natural language query")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for continuity")


class ToolCallDetail(BaseModel):
    """Record of a single tool invocation by the agent."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    execution_time_ms: float = 0.0


class AgentChatResponse(BaseModel):
    """Response from the agentic assistant with grounded, evidence-backed content."""
    conversation_id: str
    message: str = Field(..., description="Grounded response text with citations and formatting")
    tool_calls: List[ToolCallDetail] = Field(default_factory=list, description="Tools invoked and their results")
    tools_used: List[str] = Field(default_factory=list, description="Names of tools that were called")
    data: Dict[str, Any] = Field(default_factory=dict, description="Raw structured data from tool results")


class AgentToolSchema(BaseModel):
    """Schema description for a single agent tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str = "general"


# ═══════════════════════════════════════════════════════
# DASHBOARD / OVERVIEW
# ═══════════════════════════════════════════════════════

class DashboardOverview(BaseModel):
    """The main dashboard response — everything at a glance."""
    user: UserResponse
    life_state: Optional[LifeStateResponse]
    recent_transactions: List[TransactionResponse]
    upcoming_events: List[EventResponse]
    active_alerts: List[AlertResponse]
    active_goals: List[GoalResponse]
    monthly_summary: Dict[str, Any]


class WhatChangedResponse(BaseModel):
    """Signature feature: What changed since last check?"""
    changes: List[Dict[str, Any]]
    period: str  # "since_yesterday", "since_last_week"
    previous_state: Optional[LifeStateResponse]
    current_state: Optional[LifeStateResponse]


class WhatAmIForgettingResponse(BaseModel):
    """Signature feature: What am I forgetting?"""
    items: List[Dict[str, Any]]
    # Example items:
    # {"type": "expired_document", "title": "Vehicle insurance expired", "days_overdue": 15}
    # {"type": "upcoming_renewal", "title": "Health insurance renewal in 12 days", "amount": 18500}
    # {"type": "missed_service", "title": "Vehicle service overdue", "days_overdue": 45}


# ═══════════════════════════════════════════════════════
# PAGINATION
# ═══════════════════════════════════════════════════════

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    limit: int
    offset: int
    has_more: bool
