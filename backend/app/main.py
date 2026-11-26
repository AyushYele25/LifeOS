"""
LIFEOS — Personal Life Operating System

Main FastAPI application entry point.

A stateful AI decision engine that continuously understands a person's
financial and household state, detects important changes, predicts
upcoming obligations, and simulates the consequences of everyday decisions.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.config import settings
from app.database import init_db

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.DEBUG else "INFO",
)
logger.add(
    "logs/lifeos_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("🚀 Starting LIFEOS v{}", settings.APP_VERSION)

    # Initialize missing database tables safely (checkfirst=True)
    await init_db()
    logger.info("📦 Database tables initialized")

    # Seed default categories
    await _seed_default_categories()

    logger.info("✅ LIFEOS is ready")
    yield
    logger.info("👋 LIFEOS shutting down")


app = FastAPI(
    title="LIFEOS API",
    description="Personal Life Operating System — Financial & Household Intelligence Engine",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routes ──────────────────────────────────
from app.api.auth import router as auth_router
from app.api.accounts import router as accounts_router
from app.api.transactions import router as transactions_router
from app.api.dashboard import router as dashboard_router
from app.api.assets import router as assets_router
from app.api.loans import router as loans_router
from app.api.subscriptions import router as subscriptions_router
from app.api.events import router as events_router
from app.api.recurring import router as recurring_router
from app.api.alerts import router as alerts_router
from app.api.documents import router as documents_router
from app.api.lifestate import router as lifestate_router
from app.api.forecast import router as forecast_router
from app.api.risk import router as risk_router
from app.api.simulations import router as simulations_router
from app.api.rag import router as rag_router
from app.api.agent import router as agent_router
from app.api.goals import router as goals_router
from app.api.pillars import router as pillars_router

app.include_router(auth_router)
app.include_router(accounts_router)
app.include_router(transactions_router)
app.include_router(dashboard_router)
app.include_router(assets_router)
app.include_router(loans_router)
app.include_router(goals_router)
app.include_router(subscriptions_router)
app.include_router(events_router)
app.include_router(recurring_router)
app.include_router(alerts_router)
app.include_router(documents_router)
app.include_router(lifestate_router)
app.include_router(forecast_router)
app.include_router(risk_router)
app.include_router(simulations_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(pillars_router)

# Mount Web Application Static Files
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["Health"])
async def root(request: Request):
    """
    Root endpoint:
    - Returns modern interactive Web App UI when requested by web browsers (Accept: text/html).
    - Returns JSON status when requested by API clients / tests (Accept: application/json or */*).
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {
        "name": "LIFEOS",
        "version": settings.APP_VERSION,
        "status": "operational",
        "description": "Personal Life Operating System",
    }


@app.get("/app", include_in_schema=False)
async def serve_app():
    """Direct route for LIFEOS Web Application."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Web application index.html not found"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


# ─── Default Categories Seed ─────────────────────────

async def _seed_default_categories():
    """Seed system-default transaction categories if they don't exist."""
    from app.database.session import async_session
    from app.models import Category
    from sqlalchemy import select, func

    async with async_session() as session:
        # Check if categories already exist
        result = await session.execute(
            select(func.count(Category.id)).where(Category.is_system == True)
        )
        count = result.scalar()
        if count > 0:
            return

        # Default categories for Indian households
        categories = [
            # Expense categories
            {"name": "Food & Dining", "icon": "🍽️", "color": "#FF6B6B", "is_income": False},
            {"name": "Groceries", "icon": "🛒", "color": "#FF8E53", "is_income": False},
            {"name": "Shopping", "icon": "🛍️", "color": "#FEC89A", "is_income": False},
            {"name": "Transportation", "icon": "🚗", "color": "#98D8C8", "is_income": False},
            {"name": "Fuel", "icon": "⛽", "color": "#7FCDCD", "is_income": False},
            {"name": "Utilities", "icon": "💡", "color": "#A8E6CF", "is_income": False},
            {"name": "Electricity", "icon": "⚡", "color": "#FDFFAB", "is_income": False},
            {"name": "Mobile & Internet", "icon": "📱", "color": "#C4B0FF", "is_income": False},
            {"name": "Entertainment", "icon": "🎬", "color": "#B5DEFF", "is_income": False},
            {"name": "Subscriptions", "icon": "📺", "color": "#CAB8FF", "is_income": False},
            {"name": "Health & Medical", "icon": "🏥", "color": "#FF6F91", "is_income": False},
            {"name": "Insurance", "icon": "🛡️", "color": "#A0E7E5", "is_income": False},
            {"name": "Education", "icon": "📚", "color": "#FFBCBC", "is_income": False},
            {"name": "Rent", "icon": "🏠", "color": "#B4F8C8", "is_income": False},
            {"name": "EMI & Loans", "icon": "🏦", "color": "#FFE66D", "is_income": False},
            {"name": "Personal Care", "icon": "💇", "color": "#FBE7C6", "is_income": False},
            {"name": "Clothing", "icon": "👕", "color": "#E0BBE4", "is_income": False},
            {"name": "Travel", "icon": "✈️", "color": "#957DAD", "is_income": False},
            {"name": "Gifts & Donations", "icon": "🎁", "color": "#D291BC", "is_income": False},
            {"name": "Investments", "icon": "📈", "color": "#80CBC4", "is_income": False},
            {"name": "Maintenance", "icon": "🔧", "color": "#BCAAA4", "is_income": False},
            {"name": "Miscellaneous", "icon": "📌", "color": "#90A4AE", "is_income": False},
            # Income categories
            {"name": "Salary", "icon": "💰", "color": "#4CAF50", "is_income": True},
            {"name": "Freelance", "icon": "💻", "color": "#66BB6A", "is_income": True},
            {"name": "Interest", "icon": "🏦", "color": "#81C784", "is_income": True},
            {"name": "Refund", "icon": "↩️", "color": "#A5D6A7", "is_income": True},
            {"name": "Cashback", "icon": "🔄", "color": "#C8E6C9", "is_income": True},
            {"name": "Other Income", "icon": "💵", "color": "#E8F5E9", "is_income": True},
        ]

        for cat_data in categories:
            category = Category(
                name=cat_data["name"],
                icon=cat_data["icon"],
                color=cat_data["color"],
                is_system=True,
                is_income=cat_data["is_income"],
            )
            session.add(category)

        await session.commit()
        logger.info("🏷️ Seeded {} default categories", len(categories))
