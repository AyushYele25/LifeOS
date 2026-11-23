"""
LIFEOS Forecast Engine Tests

Validates:
1. Deterministic contractual floor compilation from upcoming events (loans, subscriptions, insurance).
2. Quantile ordering constraint (P10 <= P50 <= P90).
3. Indian seasonal pattern adjustments (Summer AC surge, Festive Diwali surge).
4. Backtest evaluation with MAPE and accuracy percentage.
5. Forecast API endpoints (/api/forecast, /api/forecast/backtest).
"""

from datetime import date, timedelta
from decimal import Decimal
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.models import (
    User,
    Account,
    AccountType,
    Transaction,
    TransactionType,
    Category,
    Event,
    EventType,
    EventStatus,
)
from app.services.forecast_engine import ForecastEngine


@pytest.mark.asyncio
async def test_forecast_contractual_floor_and_quantiles(db_session, test_user):
    """
    Verify contractual obligations set an immutable floor for forecasts
    and quantile confidence intervals strictly adhere to P10 <= P50 <= P90.
    """
    today = date.today()

    # 1. Add known upcoming events (Loan EMI and Insurance)
    emi_event = Event(
        user_id=test_user.id,
        title="HDFC Home Loan EMI",
        event_type=EventType.LOAN_EMI,
        event_date=today + timedelta(days=10),
        amount=Decimal("35000.00"),
        status=EventStatus.UPCOMING,
    )
    ins_event = Event(
        user_id=test_user.id,
        title="Tata AIG Car Insurance Renewal",
        event_type=EventType.INSURANCE_RENEWAL,
        event_date=today + timedelta(days=25),
        amount=Decimal("15000.00"),
        status=EventStatus.UPCOMING,
    )
    sub_event = Event(
        user_id=test_user.id,
        title="AWS Cloud Hosting",
        event_type=EventType.SUBSCRIPTION_PAYMENT,
        event_date=today + timedelta(days=45),
        amount=Decimal("4000.00"),
        status=EventStatus.UPCOMING,
    )
    db_session.add_all([emi_event, ins_event, sub_event])

    # 2. Add some historical transactions
    acc = Account(
        user_id=test_user.id,
        name="Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("100000.00"),
    )
    db_session.add(acc)
    groc_cat = (await db_session.execute(select(Category).where(Category.name == "Groceries"))).scalar_one()

    for m in range(1, 4):
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=acc.id,
                category_id=groc_cat.id,
                date=today - timedelta(days=m * 30),
                description="Groceries Supermarket",
                amount=Decimal("12000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )
    await db_session.commit()

    engine = ForecastEngine(db_session, test_user.id)
    forecast = await engine.get_forecast(as_of=today)

    # 30-day horizon check: contains EMI (35k) and Insurance (15k) = at least 50k floor
    h30 = forecast["horizon_30d"]
    assert h30["total_floor"] >= Decimal("50000.00")
    assert h30["p10_lower"] <= h30["p50_expected"]
    assert h30["p50_expected"] <= h30["p90_upper"]

    # 60-day horizon check: includes sub_event (4k)
    h60 = forecast["horizon_60d"]
    assert h60["total_floor"] >= Decimal("54000.00")
    assert h60["p10_lower"] <= h60["p50_expected"] <= h60["p90_upper"]

    # Check category quantile ordering
    for cat_item in h30["categories"]:
        assert cat_item["p10_lower"] <= cat_item["p50_expected"] <= cat_item["p90_upper"]
        assert cat_item["p50_expected"] >= cat_item["contractual_floor"]


@pytest.mark.asyncio
async def test_forecast_seasonal_adjustments(db_session, test_user):
    """
    Verify Indian seasonal adjustments (e.g. Summer AC surge in utilities, Festive shopping surge in Oct).
    """
    # Evaluate as of April 15 (Summer AC surge should trigger for utilities/electricity)
    summer_date = date(2026, 4, 15)

    acc = Account(
        user_id=test_user.id,
        name="Salary Account",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("80000.00"),
    )
    db_session.add(acc)
    util_cat = (await db_session.execute(select(Category).where(Category.name == "Utilities"))).scalar_one()

    # Historical utility expenses of ₹5,000/mo
    for m in range(1, 4):
        db_session.add(
            Transaction(
                user_id=test_user.id,
                account_id=acc.id,
                category_id=util_cat.id,
                date=summer_date - timedelta(days=m * 30),
                description="Electricity Bill",
                amount=Decimal("5000.00"),
                transaction_type=TransactionType.DEBIT,
            )
        )
    await db_session.commit()

    engine = ForecastEngine(db_session, test_user.id)
    forecast = await engine.get_forecast(as_of=summer_date)

    util_fc = next((c for c in forecast["horizon_30d"]["categories"] if c["category"] == "Utilities"), None)
    assert util_fc is not None
    # Seasonal factor should be greater than 1.0 (Summer AC surge)
    assert util_fc["seasonal_factor"] > 1.0
    # Expected forecast should reflect the surge
    assert util_fc["p50_expected"] > Decimal("5000.00")


@pytest.mark.asyncio
async def test_forecast_backtest_mape(db_session, test_user):
    """
    Verify backtesting engine calculates actual vs forecasted spend, MAPE, and accuracy percentage.
    """
    acc = Account(
        user_id=test_user.id,
        name="Savings",
        account_type=AccountType.SAVINGS,
        current_balance=Decimal("200000.00"),
    )
    db_session.add(acc)
    shop_cat = (await db_session.execute(select(Category).where(Category.name == "Shopping"))).scalar_one()

    today = date.today()
    prev_month_end = today.replace(day=1) - timedelta(days=1)
    prev_month_mid = prev_month_end.replace(day=15)

    # Add spend in previous month
    db_session.add(
        Transaction(
            user_id=test_user.id,
            account_id=acc.id,
            category_id=shop_cat.id,
            date=prev_month_mid,
            description="Lifestyle Shopping",
            amount=Decimal("25000.00"),
            transaction_type=TransactionType.DEBIT,
        )
    )
    await db_session.commit()

    engine = ForecastEngine(db_session, test_user.id)
    res = await engine.run_backtest()

    assert "evaluated_period" in res
    assert res["actual_spend"] == Decimal("25000.00")
    assert isinstance(res["mape"], float)
    assert 0.0 <= res["accuracy_pct"] <= 100.0


@pytest.mark.asyncio
async def test_forecast_api_endpoints(client: AsyncClient, auth_headers: dict):
    """
    Verify REST API endpoints:
    1. GET /api/forecast (returns 30/60/90d horizons, contractual floors, quantiles)
    2. GET /api/forecast/backtest (returns accuracy & MAPE)
    """
    # 1. GET forecast
    resp = await client.get("/api/forecast", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "horizon_30d" in data
    assert "horizon_60d" in data
    assert "horizon_90d" in data
    assert "methodology" in data
    assert "active_seasonal_factors" in data

    h30 = data["horizon_30d"]
    assert "total_floor" in h30
    assert "p10_lower" in h30
    assert "p50_expected" in h30
    assert "p90_upper" in h30
    assert len(h30["categories"]) > 0

    # 2. GET backtest
    backtest_resp = await client.get("/api/forecast/backtest", headers=auth_headers)
    assert backtest_resp.status_code == 200
    bt_data = backtest_resp.json()

    assert "evaluated_period" in bt_data
    assert "actual_spend" in bt_data
    assert "forecasted_spend" in bt_data
    assert "mape" in bt_data
    assert "accuracy_pct" in bt_data
