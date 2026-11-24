"""
LIFEOS Quick Ingestion Engine (Voice / Text & Receipt Parser)

Extracts structured transaction details from unstructured natural language
voice notes, SMS text strings, and receipt OCR text.
"""

import re
from typing import Dict, Any, Optional
from datetime import date
from app.ml.classifier import predict_category


def parse_natural_language_expense(text: str) -> Dict[str, Any]:
    """
    Parses conversational phrases like:
    - 'Spent 450 at CCD on coffee via HDFC card'
    - 'Swiggy dinner 650 rupees'
    - 'Petrol 2200 at BPCL'
    - 'Infosys salary 120000 credited today'
    - 'Paid rent 25000 to landlord'
    """
    cleaned = text.strip()
    
    # 1. Detect Transaction Type
    credit_keywords = [r"\bsalary\b", r"\bcredited\b", r"\breceived\b", r"\brefund\b", r"\bdeposit\b", r"\bcashback\b"]
    is_credit = any(re.search(pat, cleaned, re.IGNORECASE) for pat in credit_keywords)
    tx_type = "credit" if is_credit else "debit"

    # 2. Extract Amount (handles ₹450, 450.00, 450rs, 450 rupees, 1.2 lakh, 25k)
    amount = 0.0
    
    # Check for 'X lakh' or 'X k'
    lakh_match = re.search(r"(\d+(\.\d+)?)\s*(lakh|lakhs|lac|lacs)\b", cleaned, re.IGNORECASE)
    k_match = re.search(r"(\d+(\.\d+)?)\s*k\b", cleaned, re.IGNORECASE)
    
    if lakh_match:
        amount = float(lakh_match.group(1)) * 100000.0
    elif k_match:
        amount = float(k_match.group(1)) * 1000.0
    else:
        # Standard digits
        digits = re.findall(r"(?:₹|rs\.?|inr)?\s*(\d+(?:,\d+)*(?:\.\d{1,2})?)\b", cleaned, re.IGNORECASE)
        # Filter valid numbers
        valid_nums = []
        for d in digits:
            num_str = d.replace(",", "")
            try:
                num = float(num_str)
                if num > 0:
                    valid_nums.append(num)
            except ValueError:
                pass
        if valid_nums:
            # Prefer non-year numbers (e.g. not 2026) or first prominent number
            amount = next((n for n in valid_nums if n != 2026 and n != 2025 and n != 2027), valid_nums[0])

    if amount <= 0:
        amount = 100.0  # Fallback reasonable default

    # 3. Clean Description / Merchant
    # Remove amount fragments
    desc = re.sub(r"(?:₹|rs\.?|inr|rupees)\s*\d+(\.\d+)?", "", cleaned, flags=re.IGNORECASE)
    desc = re.sub(r"\b(spent|paid|for|at|on|via|credited|received|today|yesterday)\b", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+", " ", desc).strip()
    if not desc or len(desc) < 3:
        desc = cleaned

    # 4. Predict Category via ML Classifier
    ml_res = predict_category(cleaned)
    category_name = ml_res["predicted_category"]
    confidence = ml_res["confidence"]

    return {
        "raw_text": text,
        "amount": round(amount, 2),
        "description": desc.title() if len(desc) < 40 else desc[:40].title(),
        "transaction_type": tx_type,
        "predicted_category": category_name,
        "confidence": confidence,
        "date": str(date.today()),
    }


def parse_receipt_text(ocr_text: str) -> Dict[str, Any]:
    """Parses raw OCR text from a physical supermarket or restaurant bill."""
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    merchant = lines[0] if lines else "Store Receipt"
    
    # Search for total
    total = 0.0
    for line in reversed(lines):
        match = re.search(r"(?:total|net amount|amount payable|grand total)[\s:]*(?:₹|rs\.?)?\s*(\d+(?:\.\d{1,2})?)", line, re.IGNORECASE)
        if match:
            try:
                total = float(match.group(1))
                break
            except ValueError:
                pass

    if total == 0.0:
        # Fallback to largest number found
        nums = [float(n) for n in re.findall(r"\b\d+\.\d{2}\b", ocr_text)]
        total = max(nums) if nums else 250.0

    ml_res = predict_category(merchant + " " + ocr_text[:100])

    return {
        "merchant": merchant[:30],
        "total_amount": round(total, 2),
        "predicted_category": ml_res["predicted_category"],
        "confidence": ml_res["confidence"],
        "line_items_count": len(lines),
        "date": str(date.today())
    }

