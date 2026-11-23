"""
LIFEOS Document RAG API Router

Endpoints for:
1. Semantic search and grounded answer generation across user documents.
2. Source citations with page references and snippet extracts.
3. Multi-document conflict detection (e.g. conflicting renewal dates or premiums).
4. Manual and bulk reindexing of user documents.
5. Strict per-user multi-tenant data isolation.
"""

from typing import Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User, Document, DocumentType, AuditLog
from app.schemas.schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    RAGIndexResult,
    RAGReindexResult,
)
from app.services.auth_service import get_current_user
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/rag", tags=["Document RAG"])


@router.post("/query", response_model=RAGQueryResponse)
async def query_documents(
    payload: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Query user's processed documents using semantic vector search in ChromaDB.
    Returns a grounded natural language answer, source citations with page numbers,
    and automatic discrepancy detection if documents report conflicting values.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty",
        )

    response = rag_service.generate_grounded_answer(
        user_id=current_user.id,
        query=payload.query.strip(),
        top_k=payload.top_k,
        document_type=payload.document_type,
        document_id=payload.document_id,
    )

    return response


@router.post("/index/{document_id}", response_model=RAGIndexResult)
async def index_single_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Manually index or re-index a specific document into the ChromaDB vector store.
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

    if not doc.extracted_text or not doc.extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no extracted text to index",
        )

    index_res = rag_service.index_document(
        user_id=current_user.id,
        document_id=doc.id,
        filename=doc.original_filename,
        document_type=doc.document_type.value if doc.document_type else None,
        text=doc.extracted_text,
        extracted_entities=doc.extracted_entities,
    )

    user_str = str(current_user.id)
    doc.chroma_collection = rag_service.COLLECTION_NAME
    doc.chroma_ids = [f"{user_str}_{doc.id}_c{i}" for i in range(index_res.chunks_indexed)]
    await db.commit()

    return index_res


@router.delete("/index/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document_index(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Remove all vector chunks for a specific document from ChromaDB.
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

    deleted_count = rag_service.delete_document(user_id=current_user.id, document_id=doc.id)
    doc.chroma_ids = []
    await db.commit()

    return {
        "document_id": str(document_id),
        "deleted_chunks": deleted_count,
        "status": "success",
    }


@router.post("/reindex", response_model=RAGReindexResult)
async def reindex_all_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Re-index all documents for the authenticated user into ChromaDB.
    """
    return await rag_service.reindex_all_user_documents(
        user_id=current_user.id,
        db=db,
    )
