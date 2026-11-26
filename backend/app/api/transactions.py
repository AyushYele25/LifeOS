"""
LIFEOS Transaction API Routes

CRUD + CSV import + filtering for transactions.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from typing import Optional, List, Dict, Any
from datetime import date
from uuid import UUID
import csv
import io
import json
import os

from app.database import get_db
from app.models import User, Transaction, Category, AuditLog, Account
from app.models.models import TransactionType, AccountType
from app.schemas.schemas import (
    TransactionCreate, TransactionResponse, TransactionUpdate,
    TransactionFilter, PaginatedResponse, CategoryCorrectionRequest,
)
from app.services.auth_service import get_current_user
from app.ml.classifier import get_classifier
from app.config import settings

router = APIRouter(prefix="/api/transactions", tags=["Transactions"])


@router.get("", response_model=PaginatedResponse)
async def list_transactions(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    transaction_type: Optional[TransactionType] = None,
    category_id: Optional[UUID] = None,
    merchant: Optional[str] = None,
    is_recurring: Optional[bool] = None,
    is_anomaly: Optional[bool] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List transactions with powerful filtering."""
    query = select(Transaction).where(Transaction.user_id == current_user.id)

    # Apply filters
    if start_date:
        query = query.where(Transaction.date >= start_date)
    if end_date:
        query = query.where(Transaction.date <= end_date)
    if transaction_type:
        query = query.where(Transaction.transaction_type == transaction_type)
    if category_id:
        query = query.where(Transaction.category_id == category_id)
    if merchant:
        query = query.where(Transaction.merchant.ilike(f"%{merchant}%"))
    if is_recurring is not None:
        query = query.where(Transaction.is_recurring == is_recurring)
    if is_anomaly is not None:
        query = query.where(Transaction.is_anomaly == is_anomaly)
    if min_amount is not None:
        query = query.where(Transaction.amount >= min_amount)
    if max_amount is not None:
        query = query.where(Transaction.amount <= max_amount)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Fetch page
    query = query.order_by(desc(Transaction.date)).limit(limit).offset(offset)
    result = await db.execute(query)
    transactions = result.scalars().all()

    return PaginatedResponse(
        items=[TransactionResponse.model_validate(t) for t in transactions],
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total,
    )


@router.post("", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    data: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a single transaction with automatic ML categorization when category_id is omitted."""
    cat_id = data.category_id
    ml_conf = None
    is_user_classified = cat_id is not None

    if cat_id is None:
        classifier = get_classifier()
        pred_cat, conf, _ = classifier.predict(data.description)
        ml_conf = round(float(conf), 4)
        cat_res = await db.execute(
            select(Category).where(
                and_(
                    or_(Category.user_id == current_user.id, Category.is_system == True),
                    func.lower(Category.name) == pred_cat.lower(),
                )
            ).limit(1)
        )
        matched_cat = cat_res.scalar_one_or_none()
        if matched_cat:
            cat_id = matched_cat.id

    account_id = data.account_id
    target_account = None
    if account_id is not None:
        acc_res = await db.execute(
            select(Account).where(and_(Account.id == account_id, Account.user_id == current_user.id))
        )
        target_account = acc_res.scalar_one_or_none()
    else:
        acc_res = await db.execute(
            select(Account)
            .where(and_(Account.user_id == current_user.id, Account.is_active == True))
            .order_by(Account.is_primary.desc(), Account.created_at.asc())
            .limit(1)
        )
        target_account = acc_res.scalar_one_or_none()
        if target_account:
            account_id = target_account.id

    transaction = Transaction(
        user_id=current_user.id,
        date=data.date,
        description=data.description,
        amount=data.amount,
        transaction_type=data.transaction_type,
        account_id=account_id,
        category_id=cat_id,
        category_ml_confidence=ml_conf,
        is_classified_by_user=is_user_classified,
        merchant=data.merchant,
        notes=data.notes,
        source=data.source,
    )
    db.add(transaction)

    if target_account is not None and data.amount:
        bal = target_account.current_balance or 0
        if target_account.account_type == AccountType.CREDIT_CARD:
            if data.transaction_type == TransactionType.DEBIT:
                target_account.current_balance = bal + data.amount
            else:
                target_account.current_balance = bal - data.amount
        else:
            if data.transaction_type == TransactionType.CREDIT:
                target_account.current_balance = bal + data.amount
            else:
                target_account.current_balance = bal - data.amount

    await db.flush()
    await db.refresh(transaction)
    return TransactionResponse.model_validate(transaction)


@router.get("/categories")
async def list_categories(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all categories available for transaction assignment."""
    result = await db.execute(
        select(Category).where(
            or_(Category.is_system == True, Category.user_id == current_user.id)
        ).order_by(Category.name)
    )
    cats = result.scalars().all()
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "icon": c.icon,
            "color": c.color,
            "is_income": c.is_income,
        }
        for c in cats
    ]


@router.get("/classify")
async def classify_narration(
    text: str = Query(..., min_length=1),
):
    """Real-time ML classification of a transaction narration string."""
    classifier = get_classifier()
    category, confidence, is_confident = classifier.predict(text)
    return {
        "text": text,
        "predicted_category": category,
        "confidence": round(float(confidence), 4),
        "is_confident": is_confident,
    }


@router.get("/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single transaction."""
    result = await db.execute(
        select(Transaction).where(
            and_(Transaction.id == transaction_id, Transaction.user_id == current_user.id)
        )
    )
    transaction = result.scalar_one_or_none()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionResponse.model_validate(transaction)


@router.patch("/{transaction_id}", response_model=TransactionResponse)
async def update_transaction(
    transaction_id: UUID,
    data: TransactionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update transaction (category, merchant, notes, flags)."""
    result = await db.execute(
        select(Transaction).where(
            and_(Transaction.id == transaction_id, Transaction.user_id == current_user.id)
        )
    )
    transaction = result.scalar_one_or_none()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(transaction, field, value)

    # If user explicitly sets category, mark as human-classified
    if "category_id" in update_data:
        transaction.is_classified_by_user = True

    await db.flush()
    await db.refresh(transaction)
    return TransactionResponse.model_validate(transaction)


@router.delete("/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction(
    transaction_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a transaction and reverse its balance impact on any linked account."""
    result = await db.execute(
        select(Transaction).where(
            and_(Transaction.id == transaction_id, Transaction.user_id == current_user.id)
        )
    )
    transaction = result.scalar_one_or_none()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if transaction.account_id and transaction.amount:
        acc_res = await db.execute(
            select(Account).where(
                and_(Account.id == transaction.account_id, Account.user_id == current_user.id)
            )
        )
        target_account = acc_res.scalar_one_or_none()
        if target_account is not None:
            bal = target_account.current_balance or 0
            if target_account.account_type == AccountType.CREDIT_CARD:
                if transaction.transaction_type == TransactionType.DEBIT:
                    target_account.current_balance = bal - transaction.amount
                else:
                    target_account.current_balance = bal + transaction.amount
            else:
                if transaction.transaction_type == TransactionType.CREDIT:
                    target_account.current_balance = bal - transaction.amount
                else:
                    target_account.current_balance = bal + transaction.amount

    await db.delete(transaction)
    await db.flush()
    return None


@router.post("/import/preview", response_model=dict)
async def preview_csv_import(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Preview CSV import, validate columns, and predict categories on sample rows."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit")

    text = content.decode("utf-8-sig", errors="replace")
    reader = list(csv.DictReader(io.StringIO(text)))

    if not reader:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    classifier = get_classifier()
    sample_rows = []
    columns_detected = list(reader[0].keys())
    duplicate_count = 0

    for i, row in enumerate(reader[:10]):
        row_lower = {k.strip().lower(): v.strip() for k, v in row.items() if k}
        txn_date = None
        for col in ["date", "transaction_date", "txn_date", "value_date", "value date"]:
            if col in row_lower and row_lower[col]:
                txn_date = _parse_date(row_lower[col])
                break

        desc_val = ""
        for col in ["description", "narration", "particulars", "details", "remarks"]:
            if col in row_lower and row_lower[col]:
                desc_val = row_lower[col]
                break

        raw_amt = row_lower.get("amount", "") or row_lower.get("transaction_amount", "") or row_lower.get("debit", "") or row_lower.get("credit", "")
        amt = _parse_amount(raw_amt)

        if txn_date and desc_val and amt:
            pred_cat, conf, is_conf = classifier.predict(desc_val)

            # Check potential duplicate
            dup_res = await db.execute(
                select(Transaction.id).where(
                    and_(
                        Transaction.user_id == current_user.id,
                        Transaction.date == txn_date,
                        Transaction.amount == amt,
                        Transaction.description == desc_val,
                    )
                ).limit(1)
            )
            is_dup = dup_res.scalar_one_or_none() is not None
            if is_dup:
                duplicate_count += 1

            sample_rows.append({
                "row_number": i + 2,
                "date": txn_date.isoformat(),
                "description": desc_val,
                "amount": amt,
                "predicted_category": pred_cat,
                "confidence": round(conf, 3),
                "is_confident": is_conf,
                "is_duplicate": is_dup,
            })

    return {
        "filename": file.filename,
        "total_rows": len(reader),
        "columns_detected": columns_detected,
        "preview_sample": sample_rows,
        "potential_duplicates_in_sample": duplicate_count,
    }


@router.post("/import/csv", response_model=dict)
async def import_csv(
    file: UploadFile = File(...),
    account_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Import transactions from CSV with ML classification and duplicate detection.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    classifier = get_classifier()

    # Pre-cache system + user categories
    cats_res = await db.execute(
        select(Category).where(
            or_(Category.user_id == current_user.id, Category.is_system == True)
        )
    )
    category_map = {c.name.lower(): c.id for c in cats_res.scalars()}

    imported = 0
    duplicates = 0
    skipped = 0
    errors = []

    for row_num, row in enumerate(reader, start=2):
        try:
            row_lower = {k.strip().lower(): v.strip() for k, v in row.items() if k}

            # Date
            txn_date = None
            for col in ["date", "transaction_date", "txn_date", "value_date", "value date"]:
                if col in row_lower and row_lower[col]:
                    txn_date = _parse_date(row_lower[col])
                    break

            if not txn_date:
                skipped += 1
                continue

            # Description
            description = ""
            for col in ["description", "narration", "particulars", "details", "remarks"]:
                if col in row_lower and row_lower[col]:
                    description = row_lower[col]
                    break

            if not description:
                skipped += 1
                continue

            # Amount and Type
            amount = None
            txn_type = None

            debit = _parse_amount(row_lower.get("debit", "") or row_lower.get("withdrawal", ""))
            credit = _parse_amount(row_lower.get("credit", "") or row_lower.get("deposit", ""))

            if debit and debit > 0:
                amount = debit
                txn_type = TransactionType.DEBIT
            elif credit and credit > 0:
                amount = credit
                txn_type = TransactionType.CREDIT
            else:
                raw_amount = row_lower.get("amount", "") or row_lower.get("transaction_amount", "")
                amount = _parse_amount(raw_amount)
                if amount:
                    type_col = row_lower.get("type", "").lower()
                    if type_col in ["cr", "credit"]:
                        txn_type = TransactionType.CREDIT
                    elif type_col in ["dr", "debit"]:
                        txn_type = TransactionType.DEBIT
                    elif amount < 0:
                        amount = abs(amount)
                        txn_type = TransactionType.DEBIT
                    else:
                        txn_type = TransactionType.DEBIT

            if not amount or amount <= 0:
                skipped += 1
                continue

            # Duplicate check against existing user records
            dup_check = await db.execute(
                select(Transaction.id).where(
                    and_(
                        Transaction.user_id == current_user.id,
                        Transaction.date == txn_date,
                        Transaction.amount == amount,
                        Transaction.description == description,
                    )
                ).limit(1)
            )
            if dup_check.scalar_one_or_none():
                duplicates += 1
                continue

            # Run ML classification
            pred_cat, conf, is_confident = classifier.predict(description)
            cat_id = category_map.get(pred_cat.lower())
            if not cat_id:
                cat_id = category_map.get("miscellaneous")

            transaction = Transaction(
                user_id=current_user.id,
                account_id=account_id,
                category_id=cat_id,
                category_ml_confidence=round(conf, 3),
                is_classified_by_user=False,
                date=txn_date,
                description=description,
                amount=amount,
                transaction_type=txn_type,
                source="csv",
                raw_data=dict(row),
            )
            db.add(transaction)
            imported += 1

        except Exception as e:
            errors.append(f"Row {row_num}: {str(e)}")
            skipped += 1

    # Record AuditLog
    if imported > 0:
        audit = AuditLog(
            user_id=current_user.id,
            action="import_csv",
            entity_type="transaction",
            details={
                "filename": file.filename,
                "imported_count": imported,
                "duplicates_skipped": duplicates,
                "skipped_count": skipped,
            }
        )
        db.add(audit)

    await db.flush()

    return {
        "imported": imported,
        "duplicates_skipped": duplicates,
        "skipped": skipped,
        "errors": errors[:10],
        "message": f"Successfully imported {imported} transactions ({duplicates} duplicates skipped)",
    }


@router.patch("/{transaction_id}/category", response_model=TransactionResponse)
async def update_transaction_category(
    transaction_id: UUID,
    data: CategoryCorrectionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Correct the category of a transaction. Marks it as user-classified."""
    result = await db.execute(
        select(Transaction).where(
            and_(
                Transaction.id == transaction_id,
                Transaction.user_id == current_user.id,
            )
        )
    )
    txn = result.scalar_one_or_none()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    old_cat_id = str(txn.category_id) if txn.category_id else None
    txn.category_id = data.category_id
    txn.is_classified_by_user = True
    if data.notes:
        txn.notes = data.notes

    # Audit log entry for category correction
    audit = AuditLog(
        user_id=current_user.id,
        action="category_correction",
        entity_type="transaction",
        entity_id=str(txn.id),
        old_values={"category_id": old_cat_id},
        new_values={"category_id": str(data.category_id)},
        details={"notes": data.notes},
    )
    db.add(audit)

    await db.flush()
    await db.refresh(txn)
    return TransactionResponse.model_validate(txn)


@router.get("/classification/metrics", response_model=dict)
async def get_classification_metrics(
    current_user: User = Depends(get_current_user),
):
    """Get measured evaluation metrics for the transaction classification model."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    metrics_file = os.path.join(backend_dir, "ml_models", "classifier_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "unavailable",
        "message": "Model evaluation metrics not yet generated.",
    }


@router.get("/analytics/summary", response_model=dict)
async def transaction_summary(
    months: int = Query(default=1, ge=1, le=12),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get spending summary by category for the last N months."""
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=months * 30)

    result = await db.execute(
        select(
            Transaction.transaction_type,
            func.sum(Transaction.amount),
            func.count(Transaction.id),
        )
        .where(
            and_(
                Transaction.user_id == current_user.id,
                Transaction.date >= cutoff,
            )
        )
        .group_by(Transaction.transaction_type)
    )

    summary = {"total_income": 0, "total_expenses": 0, "transaction_count": 0}
    for txn_type, total, count in result:
        if txn_type == TransactionType.CREDIT:
            summary["total_income"] = float(total or 0)
        else:
            summary["total_expenses"] = float(total or 0)
        summary["transaction_count"] += count

    summary["net_savings"] = summary["total_income"] - summary["total_expenses"]
    summary["savings_rate"] = (
        (summary["net_savings"] / summary["total_income"] * 100)
        if summary["total_income"] > 0
        else 0
    )
    summary["period_months"] = months

    return summary


# ─── Helper Functions ─────────────────────────────────

def _parse_date(date_str: str) -> Optional[date]:
    """Parse various Indian date formats."""
    from datetime import datetime as dt

    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y",
        "%d %b %Y", "%d %B %Y",
        "%Y/%m/%d", "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            return dt.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_amount(amount_str: str) -> Optional[float]:
    """Parse amount string, handling Indian formatting (commas, ₹ symbol)."""
    if not amount_str:
        return None
    cleaned = amount_str.replace(",", "").replace("₹", "").replace("INR", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None
