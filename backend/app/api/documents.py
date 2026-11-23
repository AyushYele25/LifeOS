"""
LIFEOS Document Intelligence API

Endpoints for uploading documents, running entity extraction,
listing user documents, viewing extracted entities, editing metadata, and deleting documents.
Strict user data isolation is enforced on every operation.
"""

import os
import shutil
from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import (
    User,
    Document,
    DocumentType,
    AuditLog,
)
from app.schemas.schemas import (
    DocumentResponse,
    DocumentUpdate,
    DocumentProcessResult,
)
from app.services.auth_service import get_current_user
from app.services.document_service import DocumentService
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/documents", tags=["Documents"])

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))


@router.post("/upload", response_model=DocumentProcessResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a document (PDF, TXT, JSON, CSV), extract text, classify document type,
    extract structured financial entities, and automatically generate corresponding
    system records and upcoming Event obligations.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 1. Save uploaded file to disk
    file_id = uuid.uuid4()
    original_filename = file.filename or "uploaded_document"
    safe_filename = f"{current_user.id}_{file_id}_{original_filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    content_bytes = await file.read()
    file_size = len(content_bytes)

    with open(file_path, "wb") as f:
        f.write(content_bytes)

    # 2. Document Intelligence Processing
    doc_service = DocumentService()
    try:
        extracted_text = doc_service.extract_text(file_path, mime_type=file.content_type)
        doc_type, confidence = doc_service.classify_document(extracted_text, filename=original_filename)
        entities = doc_service.extract_entities(extracted_text, doc_type)

        summary = (
            f"Classified as {doc_type.value.capitalize()} with {int(confidence * 100)}% confidence. "
            f"Extracted {len(entities)} structured entities."
        )

        doc = Document(
            id=file_id,
            user_id=current_user.id,
            filename=safe_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type or "application/octet-stream",
            document_type=doc_type,
            classification_confidence=confidence,
            extracted_text=extracted_text[:10000] if extracted_text else None,  # Cap preview length
            extracted_entities=entities,
            summary=summary,
            is_processed=True,
        )
        db.add(doc)
        await db.flush()

        # 3. Synchronize extracted entities to system tables and generate obligations
        synced_records, generated_events = await doc_service.sync_entities_to_system(
            db, current_user.id, doc
        )

        event_summaries = [
            {
                "id": str(e.id),
                "event_type": e.event_type.value,
                "title": e.title,
                "event_date": str(e.event_date),
                "amount": float(e.amount) if e.amount else None,
            }
            for e in generated_events
        ]

        # Auto-index into ChromaDB Document RAG
        if doc.extracted_text and doc.extracted_text.strip():
            index_res = rag_service.index_document(
                user_id=current_user.id,
                document_id=doc.id,
                filename=doc.original_filename,
                document_type=doc.document_type.value if doc.document_type else None,
                text=doc.extracted_text,
                extracted_entities=doc.extracted_entities,
            )
            doc.chroma_collection = rag_service.COLLECTION_NAME
            doc.chroma_ids = [f"{current_user.id}_{doc.id}_c{i}" for i in range(index_res.chunks_indexed)]

        await db.commit()
        await db.refresh(doc)

        return DocumentProcessResult(
            document=doc,
            synced_records=synced_records,
            generated_events=event_summaries,
        )

    except Exception as e:
        # Save failed processing record
        doc = Document(
            id=file_id,
            user_id=current_user.id,
            filename=safe_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type,
            document_type=DocumentType.OTHER,
            is_processed=False,
            processing_error=str(e),
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        return DocumentProcessResult(
            document=doc,
            synced_records={"error": str(e)},
            generated_events=[],
        )


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    is_processed: Optional[bool] = Query(None, description="Filter by processing status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List uploaded documents for the authenticated user.
    """
    query = (
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )

    if document_type is not None:
        query = query.where(Document.document_type == document_type)
    if is_processed is not None:
        query = query.where(Document.is_processed == is_processed)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve document details, extracted text, and structured entities.
    """
    result = await db.execute(
        select(Document).where(
            and_(Document.id == document_id, Document.user_id == current_user.id)
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return doc


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: uuid.UUID,
    payload: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Correct or update document classification, extracted entities, or summary.
    """
    result = await db.execute(
        select(Document).where(
            and_(Document.id == document_id, Document.user_id == current_user.id)
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if payload.document_type is not None:
        doc.document_type = payload.document_type
    if payload.extracted_entities is not None:
        doc.extracted_entities = payload.extracted_entities
    if payload.summary is not None:
        doc.summary = payload.summary

    audit = AuditLog(
        user_id=current_user.id,
        action="document_update",
        entity_type="document",
        entity_id=str(doc.id),
        details={"updated_fields": payload.model_dump(exclude_unset=True)},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(doc)
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a document and optionally clean up the file on disk.
    """
    result = await db.execute(
        select(Document).where(
            and_(Document.id == document_id, Document.user_id == current_user.id)
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except OSError:
            pass

    # Remove from ChromaDB RAG index
    rag_service.delete_document(user_id=current_user.id, document_id=doc.id)

    await db.delete(doc)
    await db.commit()
    return None
