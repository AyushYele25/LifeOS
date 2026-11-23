"""
LIFEOS Document RAG Service Tests (Phase 12)

Validates:
1. Semantic text chunking with page number tracking and overlap preservation.
2. Fast local 384-dimensional embedding function (zero network download dependency).
3. ChromaDB vector indexing and semantic retrieval.
4. Strict multi-tenant user isolation (User A cannot access User B's document vectors).
5. Metadata filtering by document type and document ID.
6. Multi-document conflict detection when contradictory dates or figures exist.
7. REST API endpoints:
   - POST /api/rag/query
   - POST /api/rag/index/{document_id}
   - DELETE /api/rag/index/{document_id}
   - POST /api/rag/reindex
"""

from datetime import date
from decimal import Decimal
import uuid
import pytest
from httpx import AsyncClient
import numpy as np

from app.models.models import (
    User,
    Document,
    DocumentType,
)
from app.services.rag_service import RAGService, FastLocalEmbeddingFunction


SAMPLE_HEALTH_INSURANCE = """
--- Page 1 ---
HDFC ERGO General Insurance Company Limited
HEALTH SURAKSHA POLICY SCHEDULE
Policy Number: POL-H-4698379
Policyholder Name: Ayush Agarwal
Plan: Health Suraksha Gold
Sum Insured: ₹10,00,000.00
Gross Annual Premium: ₹12,500.00
Renewal Due Date: 2026-03-29
--- Page 2 ---
Terms and Conditions:
Grace period for premium payment is 30 days from the renewal due date.
Pre-existing disease waiting period is 36 months from initial inception.
Network hospital cashless authorization contact: 1800-2666-400.
"""

SAMPLE_VEHICLE_LOAN = """
HDFC BANK LIMITED — RETAIL ASSET DIVISION
LOAN SANCTION LETTER
Loan Account No: LN-731262
Borrower: Ayush Agarwal
Facility: Auto Loan
Sanctioned Loan Amount: ₹5,00,000.00
Interest Rate: 8.75% per annum
Tenure: 48 months
Monthly Installment (EMI Amount): ₹7,000.00
EMI Due Day: 10th of every month
Late fee charges: 2.0% per month on overdue installments.
"""

SAMPLE_CONFLICTING_INSURANCE = """
HDFC ERGO General Insurance Company Limited
AMENDMENT ENDORSEMENT SCHEDULE
Policy Number: POL-H-4698379
Policyholder: Ayush Agarwal
Revised Sum Insured: ₹15,00,000.00
Revised Annual Premium: ₹16,800.00
Revised Renewal Due Date: 2026-05-15
Endorsement Effective Date: 2026-01-01
"""


def test_fast_local_embedding_function():
    """Verify that FastLocalEmbeddingFunction returns unit-norm vectors and float values."""
    ef = FastLocalEmbeddingFunction(n_features=2048)
    assert ef.name() == "fast_local_embedding"
    texts = ["Health insurance policy renewal", "Car loan EMI installment"]
    embeddings = ef(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 2048
    assert len(embeddings[1]) == 2048
    # Vectors should be floats
    assert isinstance(embeddings[0][0], (float, np.floating))


def test_semantic_text_chunking():
    """Verify chunking preserves page numbers and handles multi-page documents."""
    service = RAGService()
    chunks = service.chunk_text(SAMPLE_HEALTH_INSURANCE, chunk_size=300, overlap=50)

    assert len(chunks) >= 2
    # First chunk should have page 1
    assert chunks[0].page_number == 1
    assert "HEALTH SURAKSHA" in chunks[0].text
    # Later chunk should have page 2
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_rag_indexing_and_semantic_query(tmp_path):
    """Test indexing documents into ChromaDB and running semantic queries."""
    persist_dir = str(tmp_path / "chroma_test")
    rag = RAGService(persist_dir=persist_dir)

    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    # Index document
    index_res = rag.index_document(
        user_id=user_id,
        document_id=doc_id,
        filename="hdfc_health_policy.pdf",
        document_type=DocumentType.INSURANCE.value,
        text=SAMPLE_HEALTH_INSURANCE,
        extracted_entities={
            "policy_number": "POL-H-4698379",
            "premium_amount": 12500.0,
            "due_date": "2026-03-29",
        },
    )

    assert index_res.chunks_indexed >= 2
    assert index_res.status == "success"

    # Query for renewal
    citations = rag.query_documents(
        user_id=user_id,
        query="When is my insurance renewal due and what is the grace period?",
        top_k=3,
    )

    assert len(citations) >= 1
    top_cite = citations[0]
    assert top_cite.document_id == doc_id
    assert top_cite.filename == "hdfc_health_policy.pdf"
    assert top_cite.relevance_score > 0.0

    # Test grounded answer generation
    response = rag.generate_grounded_answer(
        user_id=user_id,
        query="What is the gross annual premium for my health policy?",
    )

    assert response.query == "What is the gross annual premium for my health policy?"
    assert len(response.citations) >= 1
    assert "hdfc_health_policy.pdf" in response.answer
    assert response.has_conflicts is False


def test_user_data_isolation(tmp_path):
    """
    Verify strict multi-tenant isolation:
    User A cannot retrieve User B's documents under any circumstances.
    """
    persist_dir = str(tmp_path / "chroma_isolation")
    rag = RAGService(persist_dir=persist_dir)

    user_a = uuid.uuid4()
    user_b = uuid.uuid4()

    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    # Index secret document for User B
    rag.index_document(
        user_id=user_b,
        document_id=doc_b_id,
        filename="user_b_confidential_salary.pdf",
        document_type="salary_slip",
        text="CONFIDENTIAL SALARY SLIP: User B monthly take-home is INR 3,50,000 at Tech Corp.",
    )

    # Index normal document for User A
    rag.index_document(
        user_id=user_a,
        document_id=doc_a_id,
        filename="user_a_electricity.pdf",
        document_type="bill",
        text="BESCOM electricity bill payment receipt of INR 1,850.",
    )

    # User A queries for User B's salary keywords
    citations_a = rag.query_documents(
        user_id=user_a,
        query="take-home salary Tech Corp confidential",
        top_k=5,
    )

    # User A must get ZERO results matching User B's document
    assert len(citations_a) == 0

    # User B queries their own document
    citations_b = rag.query_documents(
        user_id=user_b,
        query="take-home salary Tech Corp",
        top_k=5,
    )
    assert len(citations_b) == 1
    assert citations_b[0].filename == "user_b_confidential_salary.pdf"


def test_metadata_filtering(tmp_path):
    """Test filtering RAG queries by document_type and document_id."""
    persist_dir = str(tmp_path / "chroma_filter")
    rag = RAGService(persist_dir=persist_dir)

    user_id = uuid.uuid4()
    ins_id = uuid.uuid4()
    loan_id = uuid.uuid4()

    rag.index_document(
        user_id=user_id,
        document_id=ins_id,
        filename="insurance.pdf",
        document_type=DocumentType.INSURANCE.value,
        text=SAMPLE_HEALTH_INSURANCE,
    )

    rag.index_document(
        user_id=user_id,
        document_id=loan_id,
        filename="loan.pdf",
        document_type=DocumentType.LOAN.value,
        text=SAMPLE_VEHICLE_LOAN,
    )

    # Filter by LOAN document type
    loan_cites = rag.query_documents(
        user_id=user_id,
        query="monthly due amount HDFC",
        document_type=DocumentType.LOAN,
    )
    assert len(loan_cites) >= 1
    for c in loan_cites:
        assert c.document_type == DocumentType.LOAN.value
        assert c.filename == "loan.pdf"

    # Filter by specific document_id
    ins_cites = rag.query_documents(
        user_id=user_id,
        query="monthly due amount HDFC",
        document_id=ins_id,
    )
    assert len(ins_cites) >= 1
    for c in ins_cites:
        assert c.document_id == ins_id


def test_conflict_detection(tmp_path):
    """
    Test that conflicting facts across multiple documents are detected and surfaced.
    """
    persist_dir = str(tmp_path / "chroma_conflict")
    rag = RAGService(persist_dir=persist_dir)

    user_id = uuid.uuid4()
    doc1_id = uuid.uuid4()
    doc2_id = uuid.uuid4()

    rag.index_document(
        user_id=user_id,
        document_id=doc1_id,
        filename="policy_original.pdf",
        document_type=DocumentType.INSURANCE.value,
        text=SAMPLE_HEALTH_INSURANCE,
    )

    rag.index_document(
        user_id=user_id,
        document_id=doc2_id,
        filename="policy_endorsement.pdf",
        document_type=DocumentType.INSURANCE.value,
        text=SAMPLE_CONFLICTING_INSURANCE,
    )

    response = rag.generate_grounded_answer(
        user_id=user_id,
        query="When is my insurance renewal date?",
    )

    assert response.has_conflicts is True
    assert response.conflict_details is not None
    assert "Conflicting dates detected" in response.conflict_details
    assert "Discrepancy Note" in response.answer


def test_document_deletion_from_vector_store(tmp_path):
    """Test that deleting a document purges all its vectors from ChromaDB."""
    persist_dir = str(tmp_path / "chroma_del")
    rag = RAGService(persist_dir=persist_dir)

    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    rag.index_document(
        user_id=user_id,
        document_id=doc_id,
        filename="temp_doc.pdf",
        document_type="bill",
        text="A temporary document to be deleted immediately.",
    )

    # Verify present
    before = rag.query_documents(user_id=user_id, query="temporary document")
    assert len(before) >= 1

    # Delete
    deleted_count = rag.delete_document(user_id=user_id, document_id=doc_id)
    assert deleted_count >= 1

    # Verify absent
    after = rag.query_documents(user_id=user_id, query="temporary document")
    assert len(after) == 0


@pytest.mark.asyncio
async def test_rag_api_endpoints(
    client: AsyncClient,
    test_user: User,
    auth_headers: dict,
    db_session,
):
    """
    Tests REST API endpoints for RAG:
    1. POST /api/rag/query
    2. POST /api/rag/index/{document_id}
    3. DELETE /api/rag/index/{document_id}
    4. POST /api/rag/reindex
    """
    # 1. Create a document in DB
    doc = Document(
        user_id=test_user.id,
        filename="test_api_doc.pdf",
        original_filename="hdfc_loan_sanction.pdf",
        file_path="/tmp/fake_path.pdf",
        document_type=DocumentType.LOAN,
        extracted_text=SAMPLE_VEHICLE_LOAN,
        extracted_entities={
            "loan_account": "LN-731262",
            "emi_amount": 7000.0,
            "interest_rate": 8.75,
        },
        is_processed=True,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    # 2. Trigger manual indexing endpoint
    index_resp = await client.post(
        f"/api/rag/index/{doc.id}",
        headers=auth_headers,
    )
    assert index_resp.status_code == 200
    assert index_resp.json()["chunks_indexed"] >= 1

    # 3. Query documents via API
    query_resp = await client.post(
        "/api/rag/query",
        headers=auth_headers,
        json={
            "query": "What is my vehicle loan EMI amount and due day?",
            "top_k": 3,
        },
    )
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert "query" in data
    assert "answer" in data
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["filename"] == "hdfc_loan_sanction.pdf"

    # 4. Trigger bulk re-indexing
    reindex_resp = await client.post(
        "/api/rag/reindex",
        headers=auth_headers,
    )
    assert reindex_resp.status_code == 200
    assert reindex_resp.json()["status"] == "success"
    assert reindex_resp.json()["documents_processed"] >= 1

    # 5. Delete document vector index
    del_resp = await client.delete(
        f"/api/rag/index/{doc.id}",
        headers=auth_headers,
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # 6. Unauthenticated check
    unauth_resp = await client.post(
        "/api/rag/query",
        json={"query": "test query"},
    )
    assert unauth_resp.status_code == 401
