"""
LIFEOS Decision Engine & Scenario Simulator

The signature "Can I afford X?" scenario engine:
Provides deterministic mathematical simulations across multiple paths:
1. Pay Cash Upfront (impact on liquid reserves & emergency runway).
2. Pay via EMI / Loan installments (interest burden, monthly cashflow, DTI impact).
3. Save Up & Delay (protecting emergency reserves, timeline to acquire).
4. New Loans, Salary Adjustments, Recurring Expenses, and Stress Scenarios.
"""

from datetime import datetime, timezone
from decimal import Decimal
import math
from typing import List, Dict, Any, Optional, Tuple
import uuid

from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Simulation,
    RiskLevel,
    AuditLog,
)
from app.services.life_state_engine import LifeStateEngine


class ScenarioSimulator:
    """
    Simulates financial decisions and evaluates resultant liquidity,
    emergency runway, DTI, and composite risk levels.
    """

    def __init__(self, db: AsyncSession, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id

    async def run_simulation(
        self,
        simulation_type: str,
        title: str,
        params: Dict[str, Any],
        persist: bool = True,
    ) -> Simulation:
        """
        Executes a deterministic simulation and optionally persists it in the simulations table.
        """
        life_engine = LifeStateEngine(self.db, self.user_id)
        current_state = await life_engine.get_current_state()

        liquid_balance = current_state.total_balance
        monthly_income = current_state.monthly_income
        monthly_expenses = current_state.monthly_expenses
        emi_burden = current_state.total_emi_burden
        runway_before = current_state.emergency_runway_months

        scenarios: List[Dict[str, Any]] = []
        recommendation: str = ""

        if simulation_type == "purchase":
            scenarios, recommendation = self._simulate_purchase(
                params=params,
                liquid_balance=liquid_balance,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                emi_burden=emi_burden,
                runway_before=runway_before,
            )
        elif simulation_type == "loan":
            scenarios, recommendation = self._simulate_loan(
                params=params,
                liquid_balance=liquid_balance,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                emi_burden=emi_burden,
                runway_before=runway_before,
            )
        elif simulation_type == "salary_change":
            scenarios, recommendation = self._simulate_salary_change(
                params=params,
                liquid_balance=liquid_balance,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                emi_burden=emi_burden,
                runway_before=runway_before,
            )
        elif simulation_type == "new_expense":
            scenarios, recommendation = self._simulate_new_expense(
                params=params,
                liquid_balance=liquid_balance,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                emi_burden=emi_burden,
                runway_before=runway_before,
            )
        elif simulation_type == "stress_test":
            scenarios, recommendation = self._simulate_stress(
                params=params,
                liquid_balance=liquid_balance,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                emi_burden=emi_burden,
                runway_before=runway_before,
            )
        else:
            raise ValueError(f"Unsupported simulation_type: {simulation_type}")

        def serialize_for_json(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_for_json(item) for item in obj]
            elif hasattr(obj, "value"):
                return obj.value
            return obj

        sanitized_params = serialize_for_json(params)
        sanitized_scenarios = serialize_for_json(scenarios)

        # Instantiate simulation record
        sim = Simulation(
            user_id=self.user_id,
            title=title,
            simulation_type=simulation_type,
            input_params=sanitized_params,
            scenarios=sanitized_scenarios,
            recommendation=recommendation,
            life_state_at_time=current_state.id,
            created_at=datetime.now(timezone.utc),
        )

        if persist:
            self.db.add(sim)
            audit = AuditLog(
                user_id=self.user_id,
                action="run_simulation",
                entity_type="simulation",
                entity_id=str(sim.id),
                details={
                    "type": simulation_type,
                    "title": title,
                    "recommendation": recommendation[:100],
                },
            )
            self.db.add(audit)
            await self.db.commit()
            await self.db.refresh(sim)

        return sim

    def _simulate_purchase(
        self,
        params: Dict[str, Any],
        liquid_balance: Decimal,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        emi_burden: Decimal,
        runway_before: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Simulates: Cash Upfront vs EMI vs Delay & Save Up.
        """
        amount = Decimal(str(params.get("amount", 0)))
        item = params.get("item", "Discretionary Purchase")
        emi_months = int(params.get("emi_months", 12))
        annual_rate = float(params.get("emi_interest_rate", 13.5))

        scenarios = []

        # 1. Path A: Cash Upfront
        bal_after_cash = liquid_balance - amount
        exp = monthly_expenses if monthly_expenses > Decimal("0.00") else Decimal("1000.00")
        runway_cash = max(0.0, float(bal_after_cash / exp)) if exp > 0 else 0.0

        if bal_after_cash < Decimal("0.00"):
            risk_cash = RiskLevel.CRITICAL
        elif runway_cash < 1.0:
            risk_cash = RiskLevel.CRITICAL
        elif runway_cash < 3.0:
            risk_cash = RiskLevel.HIGH
        elif runway_cash < 6.0:
            risk_cash = RiskLevel.MEDIUM
        else:
            risk_cash = RiskLevel.LOW

        scenarios.append({
            "name": "Pay Cash Upfront",
            "description": f"Pay ₹{amount:,.2f} immediately from liquid savings. Zero interest or financing fees.",
            "savings_after": bal_after_cash,
            "monthly_cashflow_impact": Decimal("0.00"),
            "risk_level_after": risk_cash,
            "emergency_runway_months": round(runway_cash, 1),
            "details": {
                "amount_paid": amount,
                "interest_paid": Decimal("0.00"),
                "runway_reduction_months": round(max(0.0, runway_before - runway_cash), 1),
            },
        })

        # 2. Path B: EMI Financing
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        monthly_rate = annual_rate / (12.0 * 100.0)
        n = max(1, emi_months)
        if monthly_rate > 0:
            factor = (1.0 + monthly_rate) ** n
            monthly_emi_f = float(amount) * monthly_rate * factor / (factor - 1.0)
        else:
            monthly_emi_f = float(amount) / n

        monthly_emi = Decimal(str(round(monthly_emi_f, 2)))
        total_payment = (monthly_emi * Decimal(str(n))).quantize(Decimal("0.01"))
        total_interest = max(Decimal("0.00"), total_payment - amount)

        exp_with_emi = monthly_expenses + monthly_emi
        runway_emi = max(0.0, float(liquid_balance / exp_with_emi)) if exp_with_emi > 0 else 0.0
        new_dti = float((emi_burden + monthly_emi) / monthly_income) if monthly_income > 0 else 0.0

        if new_dti > 0.50 or runway_emi < 1.0:
            risk_emi = RiskLevel.CRITICAL
        elif new_dti > 0.35 or runway_emi < 3.0:
            risk_emi = RiskLevel.HIGH
        elif runway_emi < 6.0:
            risk_emi = RiskLevel.MEDIUM
        else:
            risk_emi = RiskLevel.LOW

        scenarios.append({
            "name": f"EMI Financing ({n} Months @ {annual_rate:.1f}%)",
            "description": f"Finance over {n} months at ₹{monthly_emi:,.2f}/month. Total interest of ₹{total_interest:,.2f}.",
            "savings_after": liquid_balance,
            "monthly_cashflow_impact": -monthly_emi,
            "risk_level_after": risk_emi,
            "emergency_runway_months": round(runway_emi, 1),
            "details": {
                "monthly_emi": monthly_emi,
                "tenure_months": n,
                "interest_rate_pct": annual_rate,
                "total_interest_paid": total_interest,
                "projected_dti": round(new_dti, 4),
            },
        })

        # 3. Path C: Save Up & Delay
        monthly_surplus = max(Decimal("0.00"), monthly_income - monthly_expenses)
        if monthly_surplus > Decimal("0.00"):
            save_months = math.ceil(float(amount / monthly_surplus))
            save_desc = f"Direct all monthly savings (₹{monthly_surplus:,.2f}/mo) to acquire {item} in {save_months} months without touching reserves."
        else:
            save_months = 99
            save_desc = "Current budget has zero surplus. Reallocation or expense cuts required before purchasing."

        scenarios.append({
            "name": f"Save Up & Delay ({save_months} Months)",
            "description": save_desc,
            "savings_after": liquid_balance,
            "monthly_cashflow_impact": Decimal("0.00"),
            "risk_level_after": RiskLevel.LOW if runway_before >= 6.0 else (RiskLevel.MEDIUM if runway_before >= 3.0 else RiskLevel.HIGH),
            "emergency_runway_months": round(runway_before, 1),
            "details": {
                "months_to_save": save_months,
                "monthly_surplus_needed": monthly_surplus,
                "emergency_fund_impact": "Completely preserved",
            },
        })

        # Deterministic Recommendation
        if runway_cash >= 4.0:
            recommendation = (
                f"RECOMMENDED: Pay Cash Upfront. Your liquid balance comfortably absorbs ₹{amount:,.2f} "
                f"while retaining {runway_cash:.1f} months of emergency runway, saving ₹{total_interest:,.2f} in interest."
            )
        elif runway_cash >= 2.0 and new_dti <= 0.35 and (monthly_income - exp_with_emi) > Decimal("0.00"):
            recommendation = (
                f"RECOMMENDED: Opt for {n}-month EMI at ₹{monthly_emi:,.2f}/mo. Paying cash would compress your emergency runway "
                f"to {runway_cash:.1f} months, whereas the EMI keeps liquid reserves intact with a manageable DTI of {new_dti*100:.1f}%."
            )
        elif monthly_surplus > Decimal("0.00") and save_months <= 6:
            recommendation = (
                f"RECOMMENDED: Delay & Save Up for {save_months} months. Both cash and EMI stress your financial cushion. "
                f"Saving ₹{monthly_surplus:,.2f}/month secures the purchase without compromising debt or liquidity safety."
            )
        else:
            recommendation = (
                f"NOT RECOMMENDED AT THIS TIME. Purchasing {item} for ₹{amount:,.2f} poses severe liquidity risk "
                f"(cash runway drops to {runway_cash:.1f} months, EMI DTI reaches {new_dti*100:.1f}%). Focus on building reserves first."
            )

        return scenarios, recommendation

    def _simulate_loan(
        self,
        params: Dict[str, Any],
        liquid_balance: Decimal,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        emi_burden: Decimal,
        runway_before: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Simulates taking out a new loan (Car, Home, Personal).
        """
        principal = Decimal(str(params.get("principal", 500000)))
        tenure_months = int(params.get("tenure_months", 36))
        annual_rate = float(params.get("interest_rate", 10.5))
        loan_type = params.get("loan_type", "Personal / Vehicle Loan")

        # Base EMI calculation
        r = annual_rate / (12.0 * 100.0)
        n = tenure_months
        f = (1.0 + r) ** n
        emi_f = float(principal) * r * f / (f - 1.0)
        monthly_emi = Decimal(str(round(emi_f, 2)))
        total_repayment = (monthly_emi * Decimal(str(n))).quantize(Decimal("0.01"))
        total_interest = total_repayment - principal

        # Scenario 1: Proposed Loan
        new_dti = float((emi_burden + monthly_emi) / monthly_income) if monthly_income > 0 else 0.0
        exp_after = monthly_expenses + monthly_emi
        runway_after = max(0.0, float(liquid_balance / exp_after)) if exp_after > 0 else 0.0

        risk_1 = RiskLevel.CRITICAL if new_dti > 0.50 else (RiskLevel.HIGH if new_dti > 0.35 else (RiskLevel.MEDIUM if new_dti > 0.20 else RiskLevel.LOW))

        scenarios = [{
            "name": f"Standard Tenure ({n} Months @ {annual_rate:.1f}%)",
            "description": f"₹{monthly_emi:,.2f}/month for {n} months. Total interest of ₹{total_interest:,.2f}.",
            "savings_after": liquid_balance,
            "monthly_cashflow_impact": -monthly_emi,
            "risk_level_after": risk_1,
            "emergency_runway_months": round(runway_after, 1),
            "details": {
                "monthly_emi": monthly_emi,
                "total_interest": total_interest,
                "projected_dti": round(new_dti, 4),
            },
        }]

        # Scenario 2: Extended Tenure (+24 Months)
        n_ext = n + 24
        f_ext = (1.0 + r) ** n_ext
        emi_ext_f = float(principal) * r * f_ext / (f_ext - 1.0)
        monthly_emi_ext = Decimal(str(round(emi_ext_f, 2)))
        total_repayment_ext = (monthly_emi_ext * Decimal(str(n_ext))).quantize(Decimal("0.01"))
        total_interest_ext = total_repayment_ext - principal
        new_dti_ext = float((emi_burden + monthly_emi_ext) / monthly_income) if monthly_income > 0 else 0.0
        exp_after_ext = monthly_expenses + monthly_emi_ext
        runway_after_ext = max(0.0, float(liquid_balance / exp_after_ext)) if exp_after_ext > 0 else 0.0

        scenarios.append({
            "name": f"Extended Tenure ({n_ext} Months)",
            "description": f"Reduces monthly EMI to ₹{monthly_emi_ext:,.2f}/mo, but incurs extra interest of ₹{(total_interest_ext - total_interest):,.2f}.",
            "savings_after": liquid_balance,
            "monthly_cashflow_impact": -monthly_emi_ext,
            "risk_level_after": RiskLevel.HIGH if new_dti_ext > 0.35 else RiskLevel.MEDIUM,
            "emergency_runway_months": round(runway_after_ext, 1),
            "details": {
                "monthly_emi": monthly_emi_ext,
                "total_interest": total_interest_ext,
                "projected_dti": round(new_dti_ext, 4),
                "interest_difference": total_interest_ext - total_interest,
            },
        })

        if new_dti <= 0.35 and (monthly_income - exp_after) > Decimal("0.00"):
            recommendation = (
                f"AFFORDABLE: Proposed {loan_type} EMI of ₹{monthly_emi:,.2f}/month keeps your overall DTI at {new_dti*100:.1f}%, "
                f"well within the healthy 35% ceiling."
            )
        elif new_dti_ext <= 0.35:
            recommendation = (
                f"MARGINAL: Standard tenure pushes DTI to {new_dti*100:.1f}%. Consider extending to {n_ext} months to keep "
                f"monthly EMI at ₹{monthly_emi_ext:,.2f} and DTI at {new_dti_ext*100:.1f}%, or increase your down payment."
            )
        else:
            recommendation = (
                f"UNSAFE: This loan would push your Debt-to-Income to {new_dti*100:.1f}%, severely constraining monthly cashflow. "
                f"Not recommended without additional income or significant principal reduction."
            )

        return scenarios, recommendation

    def _simulate_salary_change(
        self,
        params: Dict[str, Any],
        liquid_balance: Decimal,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        emi_burden: Decimal,
        runway_before: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Simulates compensation adjustments (promotion, raise, or salary loss).
        """
        new_salary = Decimal(str(params.get("new_salary", float(monthly_income) * 1.15)))
        delta = new_salary - monthly_income
        new_cashflow = new_salary - monthly_expenses
        new_savings_rate = float(new_cashflow / new_salary) if new_salary > 0 else 0.0
        new_dti = float(emi_burden / new_salary) if new_salary > 0 else 0.0

        if new_salary < monthly_expenses:
            risk = RiskLevel.CRITICAL
            rec = f"WARNING: New salary of ₹{new_salary:,.2f} is below monthly living expenses (₹{monthly_expenses:,.2f}). Deficit burn will deplete reserves in {float(liquid_balance / (monthly_expenses - new_salary)):.1f} months."
        elif new_dti > 0.40:
            risk = RiskLevel.HIGH
            rec = f"CAUTION: Salary drop increases DTI to {new_dti*100:.1f}%. Prepaying debt should be prioritized."
        else:
            risk = RiskLevel.LOW
            rec = f"POSITIVE: Salary adjustment creates a healthy monthly surplus of ₹{new_cashflow:,.2f} ({new_savings_rate*100:.1f}% savings rate)."

        scenarios = [{
            "name": f"Adjusted Salary (₹{new_salary:,.2f}/mo)",
            "description": f"{'+' if delta >= 0 else ''}₹{delta:,.2f}/month net shift in baseline income.",
            "savings_after": liquid_balance,
            "monthly_cashflow_impact": delta,
            "risk_level_after": risk,
            "emergency_runway_months": round(runway_before, 1),
            "details": {
                "new_monthly_income": new_salary,
                "new_monthly_surplus": new_cashflow,
                "projected_savings_rate": round(new_savings_rate, 4),
                "projected_dti": round(new_dti, 4),
            },
        }]

        return scenarios, rec

    def _simulate_new_expense(
        self,
        params: Dict[str, Any],
        liquid_balance: Decimal,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        emi_burden: Decimal,
        runway_before: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Simulates ongoing recurring expense addition (gym, schooling, subscription).
        """
        amount = Decimal(str(params.get("amount", 2000)))
        desc = params.get("description", "Recurring Expense")
        freq = params.get("frequency", "monthly")

        monthly_cost = amount if freq == "monthly" else (amount / Decimal("12.0")).quantize(Decimal("0.01"))
        annual_cost = monthly_cost * Decimal("12.0")

        exp_after = monthly_expenses + monthly_cost
        runway_after = max(0.0, float(liquid_balance / exp_after)) if exp_after > 0 else 0.0
        new_surplus = monthly_income - exp_after

        risk = RiskLevel.CRITICAL if new_surplus < Decimal("0.00") else (RiskLevel.HIGH if runway_after < 2.0 else RiskLevel.LOW)

        rec = (
            f"AFFORDABLE: {desc} costs ₹{monthly_cost:,.2f}/mo (₹{annual_cost:,.2f}/yr). Monthly surplus remains positive at ₹{new_surplus:,.2f}."
            if new_surplus > Decimal("0.00")
            else f"DEFICIT RISK: Adding {desc} creates negative cashflow of ₹{-new_surplus:,.2f}/mo. Budget cuts required."
        )

        scenarios = [{
            "name": f"Add {desc}",
            "description": f"Incurs ₹{monthly_cost:,.2f} monthly / ₹{annual_cost:,.2f} annually.",
            "savings_after": liquid_balance,
            "monthly_cashflow_impact": -monthly_cost,
            "risk_level_after": risk,
            "emergency_runway_months": round(runway_after, 1),
            "details": {
                "annual_cost": annual_cost,
                "monthly_surplus_after": new_surplus,
            },
        }]

        return scenarios, rec

    def _simulate_stress(
        self,
        params: Dict[str, Any],
        liquid_balance: Decimal,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        emi_burden: Decimal,
        runway_before: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Simulates custom stress tests via the scenario simulator.
        """
        months = int(params.get("months", 3))
        loss_pct = float(params.get("income_reduction_pct", 100.0))

        adj_income = monthly_income * Decimal(str(1.0 - (loss_pct / 100.0)))
        burn_per_month = max(Decimal("0.00"), monthly_expenses - adj_income)
        total_drain = burn_per_month * Decimal(str(months))
        bal_after = liquid_balance - total_drain
        runway_after = max(0.0, float(bal_after / monthly_expenses)) if monthly_expenses > 0 else 0.0

        risk = RiskLevel.CRITICAL if bal_after < Decimal("0.00") else (RiskLevel.HIGH if runway_after < 2.0 else RiskLevel.MEDIUM)
        survivable = bool(bal_after >= Decimal("0.00"))

        rec = (
            f"SURVIVABLE: Liquid reserves absorb the {months}-month disruption with ₹{bal_after:,.2f} remaining ({runway_after:.1f} months runway)."
            if survivable
            else f"DEFAULT WARNING: Reserves exhausted after {float(liquid_balance / burn_per_month):.1f} months. Requires ₹{-bal_after:,.2f} external buffer."
        )

        scenarios = [{
            "name": f"{loss_pct:.0f}% Income Loss for {months} Months",
            "description": f"Cumulative liquidity drain of ₹{total_drain:,.2f}.",
            "savings_after": bal_after,
            "monthly_cashflow_impact": -(burn_per_month),
            "risk_level_after": risk,
            "emergency_runway_months": round(runway_after, 1),
            "details": {
                "total_drain": total_drain,
                "is_survivable": survivable,
            },
        }]

        return scenarios, rec

    async def get_history(self, limit: int = 20) -> List[Simulation]:
        """
        Retrieves user's historical decision simulations.
        """
        query = (
            select(Simulation)
            .where(Simulation.user_id == self.user_id)
            .order_by(desc(Simulation.created_at))
            .limit(limit)
        )
        res = await self.db.execute(query)
        return list(res.scalars().all())

    async def get_simulation(self, simulation_id: uuid.UUID) -> Optional[Simulation]:
        """
        Retrieves a single simulation by ID with user isolation.
        """
        query = select(Simulation).where(
            and_(Simulation.id == simulation_id, Simulation.user_id == self.user_id)
        )
        res = await self.db.execute(query)
        return res.scalar_one_or_none()
