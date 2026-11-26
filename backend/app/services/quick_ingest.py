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
    Parses conversational phrases & Indian bank/UPI SMS alerts like:
    - 'Spent 450 at CCD on coffee via HDFC card'
    - 'Swiggy dinner 650 rupees'
    - 'Rs.450.00 debited from A/c XX4012 on 26-11-26 to ZOMATO (UPI Ref 6331)'
    - 'INR 75,000.00 credited to HDFC A/c XX1234 Info: SALARY'
    """
    cleaned = text.strip()

    # 1. Detect Transaction Type
    credit_keywords = [r"\bsalary\b", r"\bcredited\b", r"\breceived\b", r"\brefund\b", r"\bdeposit\b", r"\bcashback\b"]
    debit_keywords = [r"\bdebited\b", r"\bspent\b", r"\bpaid\b", r"\bwithdrawn\b", r"\bpurchased\b"]
    is_credit = any(re.search(pat, cleaned, re.IGNORECASE) for pat in credit_keywords) and not any(
        re.search(r"\bdebited\b", cleaned, re.IGNORECASE) for _ in [0]
    )
    if re.search(r"\bcredited\b", cleaned, re.IGNORECASE) and not re.search(r"\bdebited\b", cleaned, re.IGNORECASE):
        is_credit = True
    elif any(re.search(pat, cleaned, re.IGNORECASE) for pat in debit_keywords) and not re.search(r"\bcredited\b", cleaned, re.IGNORECASE):
        is_credit = False
    tx_type = "credit" if is_credit else "debit"

    # Strip trailing balance clauses from bank SMS ("Avl Bal: Rs.52,140.00", "Bal Rs ...") so we extract the txn amount
    amount_text = re.split(r"\b(?:avl\.?\s*bal|avail\.?\s*bal|available\s+balance|closing\s+bal)\b", cleaned, flags=re.IGNORECASE)[0]

    # 2. Extract Amount (handles ₹450, Rs.450.00, INR 25,000, 1.2 lakh, 25k)
    amount = 0.0
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs)\b", amount_text, re.IGNORECASE)
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", amount_text, re.IGNORECASE)

    if lakh_match:
        amount = float(lakh_match.group(1)) * 100000.0
    elif k_match:
        amount = float(k_match.group(1)) * 1000.0
    else:
        currency_match = re.search(r"(?:₹|rs\.?|inr)\s*(\d+(?:,\d+)*(?:\.\d{1,2})?)\b", amount_text, re.IGNORECASE)
        if currency_match:
            amount = float(currency_match.group(1).replace(",", ""))
        else:
            digits = re.findall(r"\b(\d+(?:,\d+)*(?:\.\d{1,2})?)\b", amount_text)
            valid_nums = []
            for d in digits:
                try:
                    num = float(d.replace(",", ""))
                    if num > 0 and num not in (2024, 2025, 2026, 2027):
                        valid_nums.append(num)
                except ValueError:
                    pass
            if valid_nums:
                amount = valid_nums[0]

    if amount <= 0:
        amount = 100.0

    # 3. Clean Description / Merchant (handle both conversational phrases and Indian bank/UPI SMS)
    sms_merchant = re.search(
        r"(?:\bto\b|\bat\b|\binfo:\s*|\bvpa\s+)([A-Za-z0-9][A-Za-z0-9 &._-]{1,35}?)(?:\s*(?:\(|on\s+\d|via\s+|upi\s+ref|ref\b|avl\b|$))",
        amount_text,
        re.IGNORECASE,
    )
    if sms_merchant and re.search(r"\b(debited|credited|a/c|acct|upi\s+ref)\b", cleaned, re.IGNORECASE):
        desc = sms_merchant.group(1).strip(" -.,:")
    else:
        desc = re.sub(
            r"(?:₹|rs\.?|inr|rupees)\s*\d+(?:,\d+)*(?:\.\d{1,2})?\s*(?:k|lakhs?|lacs?)?",
            " ",
            amount_text,
            flags=re.IGNORECASE,
        )
        desc = re.sub(
            r"\b\d+(?:,\d+)*(?:\.\d{1,2})?\s*(?:k|lakhs?|lacs?|rs\.?|inr|rupees)?\b",
            " ",
            desc,
            flags=re.IGNORECASE,
        )
        desc = re.sub(
            r"\b(spent|paid|got|debited|credited|for|at|on|via|received|today|yesterday|from|to|your|a/c|acct|account)\b",
            " ",
            desc,
            flags=re.IGNORECASE,
        )
        desc = re.sub(r"\s+", " ", desc).strip(" -.,:")

    if not desc or len(desc) < 2:
        desc = cleaned[:40]

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

