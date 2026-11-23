"""
LIFEOS Financial Risk & Stress Testing Engine

Produces deterministic, multi-factor risk assessments, financial health ratios,
five-pillar risk scoring, actionable vulnerability warnings, and automated stress tests:
1. Salary/Income Loss (3 & 6 months)
2. Interest Rate Hike (+100 bps, +200 bps on loans)
3. Sudden Emergency Expense Shock (₹50k, ₹100k, ₹200k)
4. Cost of Living & Inflation Surge (+15%)
5. Custom Multi-Parameter Stress Test
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import uuid

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Account,
    AccountType,
    Transaction,
    TransactionType,
    Category,
    Loan,
    LoanStatus,
    Subscription,
    SubscriptionStatus,
    RiskLevel,
    AuditLog,
)
from app.services.life_state_engine import LifeStateEngine
from app.services.event_engine import EventEngine


class RiskEngine:
    """
    Comprehensive financial risk evaluation and deterministic stress testing engine.
    """

    def __init__(self, db: AsyncSession, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id

    async def get_assessment(self) -> Dict[str, Any]:
        """
        Calculates complete financial risk profile, ratio diagnostics,
        five-pillar risk breakdown, and standardized stress test matrix.
        """
        life_engine = LifeStateEngine(self.db, self.user_id)
        current_state = await life_engine.get_current_state()

        liquid_balance = current_state.total_balance
        monthly_income = current_state.monthly_income
        monthly_expenses = current_state.monthly_expenses
        emi_burden = current_state.total_emi_burden
        upcoming_30d = current_state.upcoming_obligations_30d
        runway_months = current_state.emergency_runway_months
        savings_rate = current_state.savings_rate

        # 1. Fetch Active Loans for interest rate sensitivity
        loan_res = await self.db.execute(
            select(Loan).where(
                and_(Loan.user_id == self.user_id, Loan.status == LoanStatus.ACTIVE)
            )
        )
        loans = loan_res.scalars().all()
        total_loan_principal = sum((l.principal_amount or Decimal("0.00") for l in loans), Decimal("0.00"))

        # 2. Fetch Category Breakdown for Concentration Risk
        expense_by_cat = current_state.expense_by_category or {}
        cat_values = sorted([float(v) for v in expense_by_cat.values()], reverse=True)
        total_exp_f = float(monthly_expenses) if monthly_expenses > 0 else 1.0
        top2_share = (sum(cat_values[:2]) / total_exp_f) if cat_values and total_exp_f > 0 else 0.0

        # 3. Calculate Core Financial Ratios
        dti_ratio = float(emi_burden / monthly_income) if monthly_income > Decimal("0.00") else (0.0 if emi_burden == Decimal("0.00") else 1.0)
        obligation_coverage = float(upcoming_30d / liquid_balance) if liquid_balance > Decimal("0.00") else (0.0 if upcoming_30d == Decimal("0.00") else 99.0)

        # 4. Five-Pillar Risk Evaluation
        breakdown: List[Dict[str, Any]] = []

        # Pillar 1: Liquidity & Runway (Weight 25%)
        if runway_months < 1.0:
            p1_score = 95.0
            p1_status = "CRITICAL"
            p1_msg = f"Critically low runway ({runway_months:.1f} months). Less than 30 days of expenses covered."
        elif runway_months < 3.0:
            p1_score = 70.0
            p1_status = "WARNING"
            p1_msg = f"Thin emergency buffer ({runway_months:.1f} months). Recommended minimum is 3 to 6 months."
        elif runway_months < 6.0:
            p1_score = 35.0
            p1_status = "OK"
            p1_msg = f"Adequate emergency buffer ({runway_months:.1f} months)."
        else:
            p1_score = 10.0
            p1_status = "OK"
            p1_msg = f"Robust emergency fund ({runway_months:.1f} months of expenses secured)."
        breakdown.append({
            "factor": "Liquidity & Emergency Runway",
            "score": p1_score,
            "weight": 0.25,
            "status": p1_status,
            "message": p1_msg,
        })

        # Pillar 2: Debt-to-Income / EMI Burden (Weight 25%)
        if dti_ratio > 0.50:
            p2_score = 90.0
            p2_status = "CRITICAL"
            p2_msg = f"Severely elevated DTI ({dti_ratio*100:.1f}%). Over half of income committed to debt servicing."
        elif dti_ratio > 0.35:
            p2_score = 65.0
            p2_status = "WARNING"
            p2_msg = f"Elevated DTI ({dti_ratio*100:.1f}%). Approaching unsafe debt thresholds."
        elif dti_ratio > 0.15:
            p2_score = 30.0
            p2_status = "OK"
            p2_msg = f"Manageable debt servicing ratio ({dti_ratio*100:.1f}%)."
        else:
            p2_score = 10.0
            p2_status = "OK"
            p2_msg = f"Low or negligible debt burden ({dti_ratio*100:.1f}% DTI)."
        breakdown.append({
            "factor": "Debt-to-Income (DTI)",
            "score": p2_score,
            "weight": 0.25,
            "status": p2_status,
            "message": p2_msg,
        })

        # Pillar 3: Cashflow & Savings Rate (Weight 20%)
        if savings_rate < 0.0:
            p3_score = 95.0
            p3_status = "CRITICAL"
            p3_msg = f"Deficit cashflow ({savings_rate*100:.1f}% savings rate). Spending regularly exceeds income."
        elif savings_rate < 0.10:
            p3_score = 65.0
            p3_status = "WARNING"
            p3_msg = f"Thin savings margin ({savings_rate*100:.1f}%). Vulnerable to unexpected costs."
        elif savings_rate < 0.25:
            p3_score = 30.0
            p3_status = "OK"
            p3_msg = f"Healthy savings rate ({savings_rate*100:.1f}%)."
        else:
            p3_score = 10.0
            p3_status = "OK"
            p3_msg = f"Strong capital accumulation rate ({savings_rate*100:.1f}%)."
        breakdown.append({
            "factor": "Cashflow & Savings Margin",
            "score": p3_score,
            "weight": 0.20,
            "status": p3_status,
            "message": p3_msg,
        })

        # Pillar 4: Short-Term Obligation Coverage (Weight 15%)
        if upcoming_30d > liquid_balance and upcoming_30d > Decimal("0.00"):
            p4_score = 95.0
            p4_status = "CRITICAL"
            p4_msg = f"30-day obligations (₹{upcoming_30d:,.2f}) exceed current liquid balance (₹{liquid_balance:,.2f})."
        elif obligation_coverage > 0.70:
            p4_score = 65.0
            p4_status = "WARNING"
            p4_msg = f"High obligation burden ({obligation_coverage*100:.1f}% of liquid funds committed within 30 days)."
        else:
            p4_score = 15.0
            p4_status = "OK"
            p4_msg = "Comfortable liquidity buffer for all upcoming 30-day obligations."
        breakdown.append({
            "factor": "Short-Term Obligation Coverage",
            "score": p4_score,
            "weight": 0.15,
            "status": p4_status,
            "message": p4_msg,
        })

        # Pillar 5: Expense Concentration & Stability (Weight 15%)
        if top2_share > 0.70 and monthly_expenses > Decimal("0.00"):
            p5_score = 65.0
            p5_status = "WARNING"
            p5_msg = f"High category concentration: top 2 categories consume {top2_share*100:.1f}% of total spend."
        else:
            p5_score = 15.0
            p5_status = "OK"
            p5_msg = "Well-diversified household expenditure profile."
        breakdown.append({
            "factor": "Expense Concentration",
            "score": p5_score,
            "weight": 0.15,
            "status": p5_status,
            "message": p5_msg,
        })

        # 5. Composite Score & Grade
        composite_score = sum(item["score"] * item["weight"] for item in breakdown)
        composite_score = round(max(0.0, min(100.0, composite_score)), 1)

        if composite_score <= 25.0:
            risk_level = RiskLevel.LOW
            risk_grade = "A"
        elif composite_score <= 45.0:
            risk_level = RiskLevel.MEDIUM
            risk_grade = "B"
        elif composite_score <= 70.0:
            risk_level = RiskLevel.HIGH
            risk_grade = "C"
        else:
            risk_level = RiskLevel.CRITICAL
            risk_grade = "D" if composite_score <= 85.0 else "F"

        # 6. Actionable Vulnerabilities & Recommendations
        vulnerabilities = []
        recommendations = []

        for item in breakdown:
            if item["status"] in ["WARNING", "CRITICAL"]:
                vulnerabilities.append(item["message"])

        if runway_months < 3.0:
            recommendations.append(
                f"Prioritize building emergency reserves to ₹{float(monthly_expenses * 3):,.2f} before entering new discretionary commitments."
            )
        if dti_ratio > 0.35:
            recommendations.append(
                "Aggressively pay down highest-interest loan principal to bring DTI below the recommended 35% threshold."
            )
        if savings_rate < 0.10:
            recommendations.append(
                "Audit recurring subscriptions and discretionary dining to restore at least a 15% monthly savings cushion."
            )
        if upcoming_30d > liquid_balance:
            recommendations.append(
                f"Immediate action needed: Transfer ₹{float(upcoming_30d - liquid_balance):,.2f} into primary savings to avoid payment default."
            )
        if not recommendations:
            recommendations.append("Financial health is solid. Maintain current savings cadence and explore index investment options.")

        # 7. Standardized Stress Test Suite
        stress_tests = self._compute_standard_stress_tests(
            liquid_balance=liquid_balance,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            emi_burden=emi_burden,
            total_principal=total_loan_principal,
        )

        return {
            "composite_risk_score": composite_score,
            "risk_level": risk_level,
            "risk_grade": risk_grade,
            "dti_ratio": round(dti_ratio, 4),
            "savings_rate": round(savings_rate, 4),
            "emergency_runway_months": runway_months,
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "breakdown": breakdown,
            "stress_tests": stress_tests,
        }

    def _compute_standard_stress_tests(
        self,
        liquid_balance: Decimal,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        emi_burden: Decimal,
        total_principal: Decimal,
    ) -> List[Dict[str, Any]]:
        """
        Runs deterministic stress tests simulating income disruptions,
        macro rate hikes, unexpected emergency shocks, and inflation.
        """
        results = []

        # Scenario 1: Salary Loss (3 Months)
        exp = monthly_expenses if monthly_expenses > Decimal("0.00") else Decimal("1000.00")
        burn_3m = exp * Decimal("3.0")
        bal_after_3m = liquid_balance - burn_3m
        runway_after_3m = max(0.0, float(bal_after_3m / exp)) if exp > 0 else 0.0

        results.append({
            "scenario_name": "Complete Income Loss (3 Months)",
            "description": "Evaluates household solvency if all income stops for 90 days while fixed and living expenses continue.",
            "monthly_cashflow_delta": -monthly_income,
            "liquid_balance_after": bal_after_3m,
            "runway_months_after": round(runway_after_3m, 1),
            "risk_level_after": RiskLevel.CRITICAL if bal_after_3m <= Decimal("0.00") else (RiskLevel.HIGH if runway_after_3m < 2.0 else RiskLevel.MEDIUM),
            "is_survivable_6_months": bool(liquid_balance >= (exp * Decimal("6.0"))),
            "details": {
                "total_outflow_3m": burn_3m,
                "deficit_amount": max(Decimal("0.00"), -bal_after_3m),
            },
        })

        # Scenario 2: Complete Income Loss (6 Months)
        burn_6m = exp * Decimal("6.0")
        bal_after_6m = liquid_balance - burn_6m
        runway_after_6m = max(0.0, float(bal_after_6m / exp)) if exp > 0 else 0.0

        results.append({
            "scenario_name": "Protracted Income Loss (6 Months)",
            "description": "Evaluates survival capacity during extended unemployment or major medical sabbatical.",
            "monthly_cashflow_delta": -monthly_income,
            "liquid_balance_after": bal_after_6m,
            "runway_months_after": round(runway_after_6m, 1),
            "risk_level_after": RiskLevel.CRITICAL if bal_after_6m <= Decimal("0.00") else RiskLevel.HIGH,
            "is_survivable_6_months": bool(bal_after_6m >= Decimal("0.00")),
            "details": {
                "total_outflow_6m": burn_6m,
                "deficit_amount": max(Decimal("0.00"), -bal_after_6m),
            },
        })

        # Scenario 3: Interest Rate Hike (+200 bps / +2.0%)
        # Approximate monthly EMI increase based on total loan principal
        # 2% annual interest on principal divided across 12 months
        monthly_emi_hike = (total_principal * Decimal("0.02") / Decimal("12.0")).quantize(Decimal("0.01"))
        new_monthly_exp = monthly_expenses + monthly_emi_hike
        new_runway = max(0.0, float(liquid_balance / new_monthly_exp)) if new_monthly_exp > 0 else 0.0
        new_dti = float((emi_burden + monthly_emi_hike) / monthly_income) if monthly_income > 0 else 0.0

        results.append({
            "scenario_name": "Interest Rate Shock (+200 bps)",
            "description": "Simulates RBI benchmark rate tightening leading to a 2% jump in floating home, auto, or personal loan rates.",
            "monthly_cashflow_delta": -monthly_emi_hike,
            "liquid_balance_after": liquid_balance,
            "runway_months_after": round(new_runway, 1),
            "risk_level_after": RiskLevel.CRITICAL if new_dti > 0.50 else (RiskLevel.HIGH if new_dti > 0.35 else RiskLevel.LOW),
            "is_survivable_6_months": True,
            "details": {
                "additional_monthly_emi": monthly_emi_hike,
                "projected_dti": round(new_dti, 4),
            },
        })

        # Scenario 4: Sudden Emergency Shock (₹1,00,000)
        shock_amt = Decimal("100000.00")
        bal_after_shock = liquid_balance - shock_amt
        runway_after_shock = max(0.0, float(bal_after_shock / exp)) if exp > 0 else 0.0

        results.append({
            "scenario_name": "Emergency Medical / Repair Shock (₹1,00,000)",
            "description": "Sudden unbudgeted expenditure (emergency hospital admission, major engine failure, urgent home repairs).",
            "monthly_cashflow_delta": -shock_amt,
            "liquid_balance_after": bal_after_shock,
            "runway_months_after": round(runway_after_shock, 1),
            "risk_level_after": RiskLevel.CRITICAL if bal_after_shock < Decimal("0.00") else (RiskLevel.HIGH if runway_after_shock < 2.0 else RiskLevel.LOW),
            "is_survivable_6_months": bool(bal_after_shock >= (exp * Decimal("6.0"))),
            "details": {
                "shock_amount": shock_amt,
                "runway_loss_months": round(max(0.0, float(liquid_balance / exp) - runway_after_shock), 1),
            },
        })

        # Scenario 5: Inflation / Cost of Living Surge (+15%)
        inflation_hike = (monthly_expenses * Decimal("0.15")).quantize(Decimal("0.01"))
        exp_inflated = monthly_expenses + inflation_hike
        runway_inflated = max(0.0, float(liquid_balance / exp_inflated)) if exp_inflated > 0 else 0.0

        results.append({
            "scenario_name": "Cost of Living & Inflation Spike (+15%)",
            "description": "Sharp escalation in food, fuel, electricity tariffs, and discretionary consumption costs.",
            "monthly_cashflow_delta": -inflation_hike,
            "liquid_balance_after": liquid_balance,
            "runway_months_after": round(runway_inflated, 1),
            "risk_level_after": RiskLevel.HIGH if runway_inflated < 2.0 else RiskLevel.LOW,
            "is_survivable_6_months": True,
            "details": {
                "extra_monthly_spend": inflation_hike,
                "adjusted_monthly_expenses": exp_inflated,
            },
        })

        return results

    async def run_custom_stress_test(
        self,
        income_drop_pct: float = 0.0,
        emergency_expense: Decimal = Decimal("0.00"),
        interest_rate_hike_bps: int = 0,
        inflation_pct: float = 0.0,
        duration_months: int = 3,
    ) -> Dict[str, Any]:
        """
        Runs a tailored stress test with combined custom shocks.
        """
        life_engine = LifeStateEngine(self.db, self.user_id)
        current_state = await life_engine.get_current_state()

        liquid_balance = current_state.total_balance
        monthly_income = current_state.monthly_income
        monthly_expenses = current_state.monthly_expenses
        emi_burden = current_state.total_emi_burden

        # 1. Fetch Active Loans for interest rate hike calculation
        loan_res = await self.db.execute(
            select(Loan).where(
                and_(Loan.user_id == self.user_id, Loan.status == LoanStatus.ACTIVE)
            )
        )
        loans = loan_res.scalars().all()
        total_loan_principal = sum((l.principal_amount or Decimal("0.00") for l in loans), Decimal("0.00"))

        # Shocks
        # A. Income drop
        adj_income = (monthly_income * Decimal(str(1.0 - (income_drop_pct / 100.0)))).quantize(Decimal("0.01"))
        # B. Interest rate hike
        rate_decimal = Decimal(str(interest_rate_hike_bps / 10000.0))
        monthly_emi_delta = (total_loan_principal * rate_decimal / Decimal("12.0")).quantize(Decimal("0.01"))
        # C. Inflation
        inflation_delta = (monthly_expenses * Decimal(str(inflation_pct / 100.0))).quantize(Decimal("0.01"))

        # Adjusted monthly expense
        adj_monthly_expenses = monthly_expenses + monthly_emi_delta + inflation_delta

        # Net monthly cashflow impact
        monthly_cashflow_before = monthly_income - monthly_expenses
        monthly_cashflow_after = adj_income - adj_monthly_expenses
        monthly_delta = monthly_cashflow_after - monthly_cashflow_before

        # Multi-month depletion
        dur_dec = Decimal(str(duration_months))
        total_monthly_burn = max(Decimal("0.00"), -monthly_cashflow_after) * dur_dec
        balance_after = liquid_balance - emergency_expense - total_monthly_burn

        runway_after = max(0.0, float(balance_after / adj_monthly_expenses)) if adj_monthly_expenses > 0 else 0.0

        if balance_after < Decimal("0.00") or runway_after < 1.0:
            risk_level = RiskLevel.CRITICAL
        elif runway_after < 3.0:
            risk_level = RiskLevel.HIGH
        elif runway_after < 6.0:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return {
            "scenario_name": f"Custom Shock: {income_drop_pct:.0f}% Income Drop, ₹{emergency_expense:,.0f} Emergency, +{interest_rate_hike_bps}bps Rate",
            "description": f"Custom {duration_months}-month simulation factoring compound income loss, emergency outlays, and macroeconomic pressure.",
            "monthly_cashflow_delta": monthly_delta,
            "liquid_balance_after": balance_after,
            "runway_months_after": round(runway_after, 1),
            "risk_level_after": risk_level,
            "is_survivable_6_months": bool(balance_after >= (adj_monthly_expenses * Decimal("6.0"))),
            "details": {
                "initial_balance": liquid_balance,
                "adjusted_monthly_income": adj_income,
                "adjusted_monthly_expenses": adj_monthly_expenses,
                "emergency_expense_deducted": emergency_expense,
                "cumulative_burn_over_duration": total_monthly_burn,
                "surplus_or_deficit": balance_after,
            },
        }
