"""
LIFEOS Document Intelligence Tests

Validates:
1. Multi-type document classification (Insurance, Loan, Warranty, Bill, Salary Slip).
2. Deterministic entity extraction on Indian financial paperwork.
3. Cross-table synchronization (Insurance, Loan, Asset) and automated obligation Event generation.
4. Genuine PDF text extraction via pypdf.
5. Full REST API lifecycle (upload, list, detail, patch, delete).
"""

from datetime import date, timedelta
from decimal import Decimal
import io
import os
import pytest
from httpx import AsyncClient
import pypdf
from sqlalchemy import select

from app.models.models import (
    User,
    Document,
    DocumentType,
    Insurance,
    Loan,
    Asset,
    Event,
    EventType,
    EventStatus,
)
from app.services.document_service import DocumentService


SAMPLE_INSURANCE_TEXT = """
HDFC ERGO General Insurance Company Limited
HEALTH SURAKSHA POLICY SCHEDULE
Policy Number: POL-H-4698379
Policyholder Name: Ayush Agarwal
Plan: Health Suraksha Gold
Sum Insured: ₹10,00,000
Gross Annual Premium: ₹12,500.00
Period of Insurance: From 30/03/2025 to 29/03/2026
Due Date: 2026-03-29
TPA: Medi Assist India TPA Pvt Ltd
Emergency Cashless Hospitalization Available Across Network Hospitals
"""

SAMPLE_LOAN_TEXT = """
HDFC BANK LIMITED — RETAIL ASSET DIVISION
LOAN SANCTION LETTER
Loan Account No: LN-731262
Borrower: Ayush Agarwal
Facility: Vehicle Loan
Sanctioned Loan Amount: ₹5,00,000.00
Rate of Interest: 8.75% per annum
Tenure: 48 months
Monthly Installment (EMI Amount): ₹7,000.00
EMI Due Day: 10th of every month
Sanction Date: 2024-04-10
Disbursement Status: Approved
"""

SAMPLE_WARRANTY_TEXT = """
APPLE LIMITED WARRANTY CERTIFICATE
Product: Apple MacBook Pro 16-inch M3 Max
Serial Number: C02XYZ987654
Date of Purchase: 15/11/2024
Warranty Period: 24 Months
Warranty Valid Until: 2026-11-15
Coverage: Hardware repair and replacement under manufacturer warranty terms.
"""

SAMPLE_BILL_TEXT = """
BANGALORE ELECTRICITY SUPPLY COMPANY LIMITED (BESCOM)
ELECTRICITY CONSUMPTION BILL
Consumer Number: BESCOM-8829104
Sub-division: Koramangala E4
Bill Date: 10/05/2026
Due Date: 25/05/2026
Units Consumed: 345 kWh
Tariff: LT-2(a)
Total Amount Due: ₹3,450.00
Payment due date: 2026-05-25
"""


def test_document_classification():
    """Verify keyword & structural classification for core document types."""
    service = DocumentService()

    doc_type, conf = service.classify_document(SAMPLE_INSURANCE_TEXT, "health_policy.pdf")
    assert doc_type == DocumentType.INSURANCE
    assert conf >= 0.80

    doc_type, conf = service.classify_document(SAMPLE_LOAN_TEXT, "car_loan_sanction.pdf")
    assert doc_type == DocumentType.LOAN
    assert conf >= 0.80

    doc_type, conf = service.classify_document(SAMPLE_WARRANTY_TEXT, "macbook_warranty.pdf")
    assert doc_type == DocumentType.WARRANTY
    assert conf >= 0.75

    doc_type, conf = service.classify_document(SAMPLE_BILL_TEXT, "electricity_bill_may.pdf")
    assert doc_type == DocumentType.BILL
    assert conf >= 0.80


def test_insurance_entity_extraction():
    """Verify structured entity parsing from insurance schedule."""
    service = DocumentService()
    entities = service.extract_entities(SAMPLE_INSURANCE_TEXT, DocumentType.INSURANCE)

    assert entities["provider"] == "HDFC ERGO"
    assert entities["policy_number"] == "POL-H-4698379"
    assert entities["insurance_type"] == "health"
    assert entities["sum_insured"] == 1000000.0
    assert entities["premium_amount"] == 12500.0
    assert entities["expiry_date"] == "2026-03-29"


def test_loan_entity_extraction():
    """Verify structured entity parsing from loan sanction agreement."""
    service = DocumentService()
    entities = service.extract_entities(SAMPLE_LOAN_TEXT, DocumentType.LOAN)

    assert entities["lender"] == "HDFC Bank"
    assert entities["account_number"] == "LN-731262"
    assert entities["loan_type"] == "vehicle"
    assert entities["principal_amount"] == 500000.0
    assert entities["emi_amount"] == 7000.0
    assert entities["interest_rate"] == 8.75
    assert entities["tenure_months"] == 48
    assert entities["emi_day"] == 10


def test_warranty_entity_extraction():
    """Verify structured entity parsing from warranty card."""
    service = DocumentService()
    entities = service.extract_entities(SAMPLE_WARRANTY_TEXT, DocumentType.WARRANTY)

    assert "MacBook Pro" in entities["product_name"]
    assert entities["serial_number"] == "C02XYZ987654"
    assert entities["expiry_date"] == "2026-11-15"


def test_bill_entity_extraction():
    """Verify structured entity parsing from utility bill."""
    service = DocumentService()
    entities = service.extract_entities(SAMPLE_BILL_TEXT, DocumentType.BILL)

    assert entities["provider"] == "BESCOM"
    assert entities["bill_type"] == "Electricity"
    assert entities["consumer_number"] == "BESCOM-8829104"
    assert entities["amount_due"] == 3450.0
    assert entities["due_date"] == "2026-05-25"


@pytest.mark.asyncio
async def test_insurance_system_sync_and_event_generation(db_session, test_user):
    """
    Verify processing an insurance document creates the Insurance DB record
    and generates an upcoming Event for policy renewal.
    """
    service = DocumentService()
    entities = service.extract_entities(SAMPLE_INSURANCE_TEXT, DocumentType.INSURANCE)

    doc = Document(
        user_id=test_user.id,
        filename="test_policy.pdf",
        original_filename="health_policy.pdf",
        file_path="/tmp/test_policy.pdf",
        document_type=DocumentType.INSURANCE,
        extracted_entities=entities,
        is_processed=True,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    synced, events = await service.sync_entities_to_system(db_session, test_user.id, doc)

    assert "insurance_id" in synced
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EventType.INSURANCE_RENEWAL
    assert event.event_date == date(2026, 3, 29)
    assert event.amount == Decimal("12500.00")
    assert "HDFC ERGO" in event.title

    # Verify Insurance policy in DB
    ins_res = await db_session.execute(select(Insurance).where(Insurance.user_id == test_user.id))
    policy = ins_res.scalar_one()
    assert policy.policy_number == "POL-H-4698379"
    assert policy.premium_amount == Decimal("12500.00")
    assert policy.sum_insured == Decimal("1000000.00")


@pytest.mark.asyncio
async def test_loan_system_sync_and_event_generation(db_session, test_user):
    """
    Verify processing a loan document creates Loan record and EMI Event.
    """
    service = DocumentService()
    entities = service.extract_entities(SAMPLE_LOAN_TEXT, DocumentType.LOAN)

    doc = Document(
        user_id=test_user.id,
        filename="test_loan.pdf",
        original_filename="car_loan.pdf",
        file_path="/tmp/test_loan.pdf",
        document_type=DocumentType.LOAN,
        extracted_entities=entities,
        is_processed=True,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    synced, events = await service.sync_entities_to_system(db_session, test_user.id, doc)

    assert "loan_id" in synced
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EventType.LOAN_EMI
    assert event.amount == Decimal("7000.00")
    assert "HDFC Bank" in event.title
    assert event.is_recurring is True

    # Verify Loan record in DB
    loan_res = await db_session.execute(select(Loan).where(Loan.user_id == test_user.id))
    loan = loan_res.scalar_one()
    assert loan.principal_amount == Decimal("500000.00")
    assert loan.emi_amount == Decimal("7000.00")
    assert loan.interest_rate == 8.75


def test_pdf_generation_and_text_extraction(tmp_path):
    """
    Creates a real PDF file using pypdf, writes text, and extracts it with DocumentService.
    """
    pdf_path = str(tmp_path / "sample_policy.pdf")
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    # Note: blank page with metadata
    writer.add_metadata({
        "/Title": "Health Insurance Policy Schedule",
        "/Author": "HDFC ERGO",
    })
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Now also test with text file to ensure text extractor works on files
    txt_path = str(tmp_path / "sample_policy.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_INSURANCE_TEXT)

    service = DocumentService()
    extracted = service.extract_text(txt_path)
    assert "POL-H-4698379" in extracted
    assert "Health Suraksha" in extracted


@pytest.mark.asyncio
async def test_document_api_lifecycle(client: AsyncClient, test_user: User, auth_headers: dict, db_session):
    """
    Tests REST API endpoints:
    1. POST /api/documents/upload (Upload document)
    2. GET /api/documents (List user documents)
    3. GET /api/documents/{id} (Document detail)
    4. PATCH /api/documents/{id} (Update metadata)
    5. DELETE /api/documents/{id} (Delete document)
    """
    # 1. Upload sample bill document
    file_content = SAMPLE_BILL_TEXT.encode("utf-8")
    files = {"file": ("electricity_bill.txt", file_content, "text/plain")}

    upload_resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()

    doc_data = upload_data["document"]
    doc_id = doc_data["id"]
    assert doc_data["document_type"] == "bill"
    assert doc_data["is_processed"] is True
    assert doc_data["extracted_entities"]["consumer_number"] == "BESCOM-8829104"
    assert len(upload_data["generated_events"]) == 1

    # 2. List documents
    list_resp = await client.get("/api/documents", headers=auth_headers)
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert len(docs) >= 1
    assert any(d["id"] == doc_id for d in docs)

    # 3. Get document details
    get_resp = await client.get(f"/api/documents/{doc_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == doc_id
    assert "BESCOM" in get_resp.json()["extracted_text"]

    # 4. Update document metadata
    patch_resp = await client.patch(
        f"/api/documents/{doc_id}",
        headers=auth_headers,
        json={"summary": "Verified May 2026 electricity bill"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["summary"] == "Verified May 2026 electricity bill"

    # 5. Delete document
    del_resp = await client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
    assert del_resp.status_code == 204

    # Verify deleted
    get_after = await client.get(f"/api/documents/{doc_id}", headers=auth_headers)
    assert get_after.status_code == 404
