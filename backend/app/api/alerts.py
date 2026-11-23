"""
LIFEOS Alerts API

Endpoints for retrieving, managing, and running anomaly detection alerts.
Per-user isolation is strictly enforced.
"""

from typing import List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import (
    User,
    Alert,
    AlertType,
    AlertSeverity,
    AuditLog,
)
from app.schemas.schemas import (
    AlertResponse,
    AlertDismissRequest,
    AlertSummaryResponse,
    AnomalyDetectionRunResponse,
)
from app.services.auth_service import get_current_user
from app.services.anomaly_detector import AnomalyDetector

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    is_dismissed: Optional[bool] = Query(None, description="Filter by dismissed status"),
    is_read: Optional[bool] = Query(None, description="Filter by read status"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    alert_type: Optional[AlertType] = Query(None, description="Filter by alert type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List alerts for the authenticated user with optional filtering.
    """
    query = (
        select(Alert)
        .where(Alert.user_id == current_user.id)
        .order_by(Alert.created_at.desc())
    )

    if is_dismissed is not None:
        query = query.where(Alert.is_dismissed == is_dismissed)
    if is_read is not None:
        query = query.where(Alert.is_read == is_read)
    if severity is not None:
        query = query.where(Alert.severity == severity)
    if alert_type is not None:
        query = query.where(Alert.alert_type == alert_type)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    alerts = result.scalars().all()
    return alerts


@router.get("/summary", response_model=AlertSummaryResponse)
async def get_alert_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get aggregated counts of active (undismissed) alerts by severity and top recent alerts.
    """
    base_filter = and_(Alert.user_id == current_user.id, Alert.is_dismissed == False)

    # Active count
    total_res = await db.execute(select(func.count(Alert.id)).where(base_filter))
    total_active = total_res.scalar() or 0

    # Severity counts
    crit_res = await db.execute(
        select(func.count(Alert.id)).where(
            and_(base_filter, Alert.severity == AlertSeverity.CRITICAL)
        )
    )
    critical_count = crit_res.scalar() or 0

    warn_res = await db.execute(
        select(func.count(Alert.id)).where(
            and_(base_filter, Alert.severity == AlertSeverity.WARNING)
        )
    )
    warning_count = warn_res.scalar() or 0

    info_res = await db.execute(
        select(func.count(Alert.id)).where(
            and_(base_filter, Alert.severity == AlertSeverity.INFO)
        )
    )
    info_count = info_res.scalar() or 0

    # Recent active alerts
    recent_res = await db.execute(
        select(Alert)
        .where(base_filter)
        .order_by(Alert.created_at.desc())
        .limit(5)
    )
    recent_alerts = recent_res.scalars().all()

    return AlertSummaryResponse(
        total_active=total_active,
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
        recent_alerts=recent_alerts,
    )


@router.post("/run-detection", response_model=AnomalyDetectionRunResponse)
async def run_anomaly_detection(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Trigger the hybrid statistical + ML anomaly detector across the user's transactions.
    """
    detector = AnomalyDetector(db, current_user.id)
    report = await detector.run_detection()

    return AnomalyDetectionRunResponse(
        scanned_transactions=report["scanned_transactions"],
        anomalies_detected=report["anomalies_detected"],
        new_alerts_created=report["new_alerts_created"],
        alerts=report["alerts"],
    )


@router.patch("/{alert_id}/read", response_model=AlertResponse)
async def mark_alert_read(
    alert_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Mark an alert as read.
    """
    result = await db.execute(
        select(Alert).where(and_(Alert.id == alert_id, Alert.user_id == current_user.id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    alert.is_read = True
    await db.commit()
    await db.refresh(alert)
    return alert


@router.patch("/{alert_id}/dismiss", response_model=AlertResponse)
async def dismiss_alert(
    alert_id: uuid.UUID,
    payload: Optional[AlertDismissRequest] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Dismiss an alert, with optional user feedback (e.g. mark as expected).
    """
    result = await db.execute(
        select(Alert).where(and_(Alert.id == alert_id, Alert.user_id == current_user.id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    alert.is_dismissed = True
    metadata = dict(alert.metadata_json or {})

    if payload:
        if payload.mark_as_expected:
            metadata["marked_as_expected_by_user"] = True
        if payload.feedback:
            metadata["user_feedback"] = payload.feedback
        alert.metadata_json = metadata

    audit = AuditLog(
        user_id=current_user.id,
        action="alert_dismiss",
        entity_type="alert",
        entity_id=str(alert.id),
        details={
            "alert_type": alert.alert_type,
            "marked_as_expected": payload.mark_as_expected if payload else False,
            "feedback": payload.feedback if payload else None,
        },
    )
    db.add(audit)
    await db.commit()
    await db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Permanently delete an alert.
    """
    result = await db.execute(
        select(Alert).where(and_(Alert.id == alert_id, Alert.user_id == current_user.id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    await db.delete(alert)
    await db.commit()
    return None
