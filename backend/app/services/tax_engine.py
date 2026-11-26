"""
LIFEOS Indian Tax & Salary Structuring Copilot

Compares Old vs. New Income Tax Regimes (FY 2025-26 / FY 2026-27)
Incorporates: Standard Deduction, HRA Exemption, Section 80C, 80D, 80CCD(1B) NPS, and Home Loan Sec 24.
"""

from typing import Dict, Any


def calculate_new_regime_tax(gross_income: float) -> Dict[str, Any]:
    """
    New Tax Regime (Section 115BAC - FY 2025-26 / 2026-27 Budget 2025):
    - Salaried Standard Deduction: ₹75,000
    - Slabs:
      0 - 4L: 0%
      4L - 8L: 5%
      8L - 12L: 10%
      12L - 16L: 15%
      16L - 20L: 20%
      20L - 24L: 25%
      > 24L: 30%
    - Sec 87A Rebate: Full rebate up to ₹12,00,000 taxable income (Gross <= ₹12,75,000),
      with marginal relief right above ₹12,00,000 so tax increase never exceeds excess income.
    - 4% Health & Education Cess.
    """
    standard_deduction = 75000.0
    taxable_income = max(0.0, gross_income - standard_deduction)

    if taxable_income <= 1200000.0:
        return {
            "regime": "New Regime",
            "gross_income": gross_income,
            "deductions_claimed": standard_deduction,
            "taxable_income": taxable_income,
            "base_tax": 0.0,
            "rebate_87a": 0.0,
            "cess_4pct": 0.0,
            "total_tax_payable": 0.0,
            "effective_tax_rate": 0.0
        }

    slabs = [
        (400000.0, 800000.0, 0.05),
        (800000.0, 1200000.0, 0.10),
        (1200000.0, 1600000.0, 0.15),
        (1600000.0, 2000000.0, 0.20),
        (2000000.0, 2400000.0, 0.25),
        (2400000.0, float("inf"), 0.30),
    ]
    base_tax = 0.0
    for low, high, rate in slabs:
        if taxable_income > low:
            base_tax += (min(taxable_income, high) - low) * rate

    # Section 87A marginal relief: tax before cess cannot exceed income above ₹12,00,000
    excess_over_rebate = taxable_income - 1200000.0
    if base_tax > excess_over_rebate:
        base_tax = excess_over_rebate

    cess = base_tax * 0.04
    total_tax = base_tax + cess

    return {
        "regime": "New Regime",
        "gross_income": gross_income,
        "deductions_claimed": standard_deduction,
        "taxable_income": taxable_income,
        "base_tax": round(base_tax, 2),
        "rebate_87a": 0.0,
        "cess_4pct": round(cess, 2),
        "total_tax_payable": round(total_tax, 2),
        "effective_tax_rate": round((total_tax / gross_income) * 100, 2) if gross_income > 0 else 0.0
    }


def calculate_old_regime_tax(
    gross_income: float,
    hra_exemption: float = 0.0,
    sec_80c: float = 150000.0,
    sec_80d: float = 25000.0,
    home_loan_sec24: float = 0.0,
    sec_80ccd_nps: float = 50000.0,
) -> Dict[str, Any]:
    """
    Old Tax Regime:
    - Salaried Standard Deduction: ₹50,000
    - HRA Exemption Sec 10(13A)
    - Sec 80C: capped at ₹1,50,000
    - Sec 80D: capped at ₹25,000 / ₹50,000
    - Sec 24(b) Home loan interest: capped at ₹2,00,000
    - Sec 80CCD(1B) NPS: capped at ₹50,000
    - Slabs:
      0 - 2.5L: 0%
      2.5L - 5L: 5%
      5L - 10L: 20%
      > 10L: 30%
    - Sec 87A rebate up to ₹5,00,000 taxable income.
    - 4% Cess.
    """
    standard_deduction = 50000.0
    c_capped = min(150000.0, max(0.0, sec_80c))
    d_capped = min(75000.0, max(0.0, sec_80d))
    sec24_capped = min(200000.0, max(0.0, home_loan_sec24))
    nps_capped = min(50000.0, max(0.0, sec_80ccd_nps))
    hra_capped = max(0.0, hra_exemption)

    total_deductions = standard_deduction + c_capped + d_capped + sec24_capped + nps_capped + hra_capped
    taxable_income = max(0.0, gross_income - total_deductions)

    if taxable_income <= 500000.0:
        return {
            "regime": "Old Regime",
            "gross_income": gross_income,
            "deductions_breakdown": {
                "standard_deduction": standard_deduction,
                "sec_80c": c_capped,
                "sec_80d": d_capped,
                "sec_24_home_loan": sec24_capped,
                "sec_80ccd_nps": nps_capped,
                "hra_exemption": hra_capped
            },
            "total_deductions": total_deductions,
            "taxable_income": taxable_income,
            "base_tax": 0.0,
            "rebate_87a": 0.0,
            "cess_4pct": 0.0,
            "total_tax_payable": 0.0,
            "effective_tax_rate": 0.0
        }

    base_tax = 0.0
    # 2.5L - 5L @ 5% = 12,500
    if taxable_income > 250000.0:
        slab = min(taxable_income, 500000.0) - 250000.0
        base_tax += slab * 0.05
    # 5L - 10L @ 20% = up to 1,00,000
    if taxable_income > 500000.0:
        slab = min(taxable_income, 1000000.0) - 500000.0
        base_tax += slab * 0.20
    # > 10L @ 30%
    if taxable_income > 1000000.0:
        slab = taxable_income - 1000000.0
        base_tax += slab * 0.30

    cess = base_tax * 0.04
    total_tax = base_tax + cess

    return {
        "regime": "Old Regime",
        "gross_income": gross_income,
        "deductions_breakdown": {
            "standard_deduction": standard_deduction,
            "sec_80c": c_capped,
            "sec_80d": d_capped,
            "sec_24_home_loan": sec24_capped,
            "sec_80ccd_nps": nps_capped,
            "hra_exemption": hra_capped
        },
        "total_deductions": total_deductions,
        "taxable_income": taxable_income,
        "base_tax": round(base_tax, 2),
        "rebate_87a": 0.0,
        "cess_4pct": round(cess, 2),
        "total_tax_payable": round(total_tax, 2),
        "effective_tax_rate": round((total_tax / gross_income) * 100, 2) if gross_income > 0 else 0.0
    }


def compare_tax_regimes(
    gross_income: float,
    hra_exemption: float = 0.0,
    sec_80c: float = 150000.0,
    sec_80d: float = 25000.0,
    home_loan_sec24: float = 0.0,
    sec_80ccd_nps: float = 50000.0,
) -> Dict[str, Any]:
    """Compares Old vs New Regime and recommends the best option."""
    new_res = calculate_new_regime_tax(gross_income)
    old_res = calculate_old_regime_tax(
        gross_income, hra_exemption, sec_80c, sec_80d, home_loan_sec24, sec_80ccd_nps
    )

    new_tax = new_res["total_tax_payable"]
    old_tax = old_res["total_tax_payable"]
    diff = abs(new_tax - old_tax)

    if old_tax < new_tax:
        recommendation = f"Choose OLD REGIME. You save ₹{diff:,.0f} in taxes due to your HRA, 80C, and NPS deductions."
        winner = "Old Regime"
        savings = diff
    else:
        recommendation = f"Choose NEW REGIME. You save ₹{diff:,.0f} in taxes with zero investment lock-in and minimal paperwork."
        winner = "New Regime"
        savings = diff

    return {
        "gross_income": gross_income,
        "winner": winner,
        "annual_tax_savings": round(savings, 2),
        "monthly_in_hand_boost": round(savings / 12.0, 2),
        "recommendation": recommendation,
        "new_regime": new_res,
        "old_regime": old_res
    }
