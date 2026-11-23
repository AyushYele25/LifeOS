"""
LIFEOS Document Intelligence Service

Handles:
1. Multi-format text extraction (PDF via pypdf, TXT, JSON, Markdown).
2. Rule-based & heuristic document classification (Insurance, Loan, Warranty, Bill, Salary Slip, Receipt).
3. Deterministic entity extraction using pattern matching & regex tailored for Indian financial documents.
4. Cross-table synchronization (Insurance policies, Loans, Assets, and Future Events).
"""

from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
import io
import os
import re
from typing import Dict, Any, Optional, Tuple, List
import uuid

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import pypdf

from app.models.models import (
    Document,
    DocumentType,
    Insurance,
    InsuranceType,
    SubscriptionFrequency,
    Loan,
    LoanType,
    LoanStatus,
    Asset,
    AssetType,
    Event,
    EventType,
    EventStatus,
    AuditLog,
)


class DocumentService:
    """
    Core document processing and entity extraction service.
    """

    KNOWN_INSURERS = [
        "HDFC ERGO", "Star Health", "ICICI Lombard", "Care Health",
        "Niva Bupa", "Max Bupa", "Tata AIG", "Bajaj Allianz",
        "SBI General", "LIC", "New India Assurance", "Acko",
    ]

    KNOWN_LENDERS = [
        "HDFC Bank", "State Bank of India", "SBI", "ICICI Bank",
        "Axis Bank", "Kotak Mahindra Bank", "Bank of Baroda",
        "Punjab National Bank", "Bajaj Finserv", "Tata Capital",
    ]

    KNOWN_UTILITY_PROVIDERS = [
        "BESCOM", "TNEB", "Tata Power", "Adani Electricity", "MSEDCL",
        "BSES Yamuna", "BSES Rajdhani", "Airtel", "Jio", "JioFiber", "ACT Fibernet",
        "Indraprastha Gas", "Mahanagar Gas", "BWSSB",
    ]

    def extract_text(self, file_path: str, mime_type: Optional[str] = None) -> str:
        """
        Extracts plain text content from a PDF or text-based document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found at: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Handle PDF
        if ext == ".pdf" or mime_type == "application/pdf":
            try:
                reader = pypdf.PdfReader(file_path)
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
                extracted = "\n".join(pages_text).strip()
                if extracted:
                    return extracted
            except Exception as e:
                # If PDF reading failed, fallback to binary-text salvage
                pass

        # Handle TXT, JSON, CSV, MD
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        except Exception:
            with open(file_path, "r", encoding="latin-1", errors="replace") as f:
                return f.read().strip()

    def classify_document(self, text: str, filename: str = "") -> Tuple[DocumentType, float]:
        """
        Classifies the document into a DocumentType with an associated confidence score.
        """
        content = (filename + " " + text).lower()

        scores: Dict[DocumentType, int] = {
            DocumentType.INSURANCE: 0,
            DocumentType.LOAN: 0,
            DocumentType.WARRANTY: 0,
            DocumentType.BILL: 0,
            DocumentType.SALARY_SLIP: 0,
            DocumentType.RECEIPT: 0,
        }

        # Insurance indicators
        insurance_keywords = [
            "policy schedule", "policy number", "sum insured", "sum assured",
            "premium amount", "health insurance", "motor insurance", "tpa",
            "cashless hospital", "hdfc ergo", "star health", "care health",
            "mediclaim", "policyholder", "insurance policy"
        ]
        for kw in insurance_keywords:
            if kw in content:
                scores[DocumentType.INSURANCE] += 2

        # Loan indicators
        loan_keywords = [
            "sanction letter", "loan agreement", "principal amount", "loan account",
            "emi amount", "monthly installment", "rate of interest", "roi",
            "tenure in months", "repayment schedule", "lender", "borrower", "vehicle loan", "home loan"
        ]
        for kw in loan_keywords:
            if kw in content:
                scores[DocumentType.LOAN] += 2

        # Warranty indicators
        warranty_keywords = [
            "warranty card", "warranty period", "guarantee certificate", "limited warranty",
            "warranty valid", "serial number", "coverage period", "extended warranty",
            "manufacturer warranty", "warranty terms"
        ]
        for kw in warranty_keywords:
            if kw in content:
                scores[DocumentType.WARRANTY] += 2

        # Bill indicators
        bill_keywords = [
            "electricity bill", "water bill", "broadband bill", "power distribution",
            "bescom", "tneb", "consumer no", "consumer number", "amount due",
            "bill date", "due date", "tariff", "units consumed", "kwh", "discom"
        ]
        for kw in bill_keywords:
            if kw in content:
                scores[DocumentType.BILL] += 2

        # Salary slip indicators
        salary_keywords = [
            "payslip", "salary slip", "earnings", "basic pay", "hra",
            "provident fund", "pf deduction", "net pay", "gross earnings", "ctc"
        ]
        for kw in salary_keywords:
            if kw in content:
                scores[DocumentType.SALARY_SLIP] += 2

        # Receipt indicators
        receipt_keywords = [
            "tax invoice", "invoice no", "gstin", "subtotal", "grand total",
            "payment receipt", "pos transaction", "order id"
        ]
        for kw in receipt_keywords:
            if kw in content:
                scores[DocumentType.RECEIPT] += 2

        best_type, max_score = max(scores.items(), key=lambda item: item[1])

        if max_score >= 6:
            confidence = min(0.98, 0.70 + (max_score * 0.03))
            return best_type, round(confidence, 2)
        elif max_score >= 2:
            return best_type, 0.70
        else:
            return DocumentType.OTHER, 0.40

    def extract_entities(self, text: str, doc_type: DocumentType) -> Dict[str, Any]:
        """
        Extracts structured fields matching the document type schema.
        """
        entities: Dict[str, Any] = {}

        if doc_type == DocumentType.INSURANCE:
            entities = self._extract_insurance_entities(text)
        elif doc_type == DocumentType.LOAN:
            entities = self._extract_loan_entities(text)
        elif doc_type == DocumentType.WARRANTY:
            entities = self._extract_warranty_entities(text)
        elif doc_type == DocumentType.BILL:
            entities = self._extract_bill_entities(text)
        elif doc_type == DocumentType.SALARY_SLIP:
            entities = self._extract_salary_entities(text)
        else:
            entities = self._extract_generic_entities(text)

        return entities

    def _extract_date(self, text: str, patterns: List[str]) -> Optional[str]:
        """Helper to find the first matching date string and format to YYYY-MM-DD."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                # Try parsing standard formats
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y"):
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
        return None

    def _extract_amount(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to find the first matching monetary amount."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amt_str = match.group(1).replace(",", "").strip()
                try:
                    return float(amt_str)
                except ValueError:
                    continue
        return None

    def _extract_insurance_entities(self, text: str) -> Dict[str, Any]:
        """Extracts insurance policy details."""
        entities: Dict[str, Any] = {}

        # 1. Provider
        for insurer in self.KNOWN_INSURERS:
            if re.search(r"\b" + re.escape(insurer) + r"\b", text, re.IGNORECASE):
                entities["provider"] = insurer
                break
        if "provider" not in entities:
            m = re.search(r"(?:insurer|company|provider)[\s:]*([A-Za-z\s&]{3,40})", text, re.IGNORECASE)
            if m:
                entities["provider"] = m.group(1).strip()

        # 2. Policy Number
        pol_m = re.search(
            r"(?:policy\s*(?:no|num|number|id)\b[\s:#\.\-]+)([A-Z0-9\-\/]{5,30})",
            text, re.IGNORECASE
        )
        if not pol_m:
            pol_m = re.search(r"(?:policy\s*[:#]+)\s*([A-Z0-9\-\/]{5,30})", text, re.IGNORECASE)
        if pol_m:
            entities["policy_number"] = pol_m.group(1).strip()

        # 3. Policy Type
        lower_text = text.lower()
        if "health" in lower_text or "mediclaim" in lower_text:
            entities["insurance_type"] = "health"
        elif "motor" in lower_text or "vehicle" in lower_text or "car" in lower_text or "bike" in lower_text:
            entities["insurance_type"] = "vehicle"
        elif "term" in lower_text or "life" in lower_text:
            entities["insurance_type"] = "life"
        else:
            entities["insurance_type"] = "other"

        # 4. Sum Insured
        sum_ins = self._extract_amount(text, [
            r"(?:sum\s*insured|sum\s*assured|coverage\s*amount)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?:coverage|cover)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if sum_ins:
            entities["sum_insured"] = sum_ins

        # 5. Premium Amount
        premium = self._extract_amount(text, [
            r"(?:annual\s*premium|total\s*premium|premium\s*amount|gross\s*premium|gross\s*annual\s*premium)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?:premium)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if premium:
            entities["premium_amount"] = premium

        # 6. Expiry / Renewal Date
        expiry = self._extract_date(text, [
            r"(?:expiry\s*date|valid\s*upto|renewal\s*date|due\s*date|end\s*date)[\s:]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ])
        if expiry:
            entities["expiry_date"] = expiry
            entities["renewal_date"] = expiry

        return entities

    def _extract_loan_entities(self, text: str) -> Dict[str, Any]:
        """Extracts loan sanction details."""
        entities: Dict[str, Any] = {}

        # 1. Lender
        for lender in self.KNOWN_LENDERS:
            if re.search(r"\b" + re.escape(lender) + r"\b", text, re.IGNORECASE):
                entities["lender"] = lender
                break
        if "lender" not in entities:
            m = re.search(r"(?:lender|bank|institution)[\s:]*([A-Za-z\s&]{3,40})", text, re.IGNORECASE)
            if m:
                entities["lender"] = m.group(1).strip()

        # 2. Account / Reference Number
        acc_m = re.search(
            r"(?:loan\s*(?:a\/c|account|acc|no|num|number|ref)\b[\s:#\.\-]+)([A-Z0-9\-\/]{5,30})",
            text, re.IGNORECASE
        )
        if not acc_m:
            acc_m = re.search(r"(?:account\s*(?:no|number|num)\b[\s:#\.\-]+)([A-Z0-9\-\/]{5,30})", text, re.IGNORECASE)
        if acc_m:
            entities["account_number"] = acc_m.group(1).strip()

        # 3. Loan Type
        lower = text.lower()
        if "vehicle" in lower or "car" in lower or "auto" in lower:
            entities["loan_type"] = "vehicle"
        elif "home" in lower or "housing" in lower:
            entities["loan_type"] = "home"
        elif "personal" in lower:
            entities["loan_type"] = "personal"
        elif "education" in lower:
            entities["loan_type"] = "education"
        else:
            entities["loan_type"] = "other"

        # 4. Principal Amount
        principal = self._extract_amount(text, [
            r"(?:principal\s*amount|sanctioned\s*amount|loan\s*amount|sanctioned\s*loan\s*amount)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if principal:
            entities["principal_amount"] = principal

        # 5. EMI Amount
        emi = self._extract_amount(text, [
            r"(?:monthly\s*installment\s*\([^\)]*emi[^\)]*\))[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?:emi\s*amount|monthly\s*installment|monthly\s*emi)[\s:\(\)]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?:emi)[\s:\(\)]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if emi:
            entities["emi_amount"] = emi

        # 6. Interest Rate
        roi_m = re.search(r"(?:interest\s*rate|roi|rate\s*of\s*interest)[\s:]*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.IGNORECASE)
        if roi_m:
            entities["interest_rate"] = float(roi_m.group(1))

        # 7. Tenure Months
        tenure_m = re.search(r"(?:tenure|duration|period)[\s:]*([0-9]+)\s*(?:months|mths)?", text, re.IGNORECASE)
        if tenure_m:
            entities["tenure_months"] = int(tenure_m.group(1))

        # 8. EMI Due Day
        due_day_m = re.search(r"(?:emi\s*due\s*day|due\s*day|day\s*of\s*month)[\s:]*([0-9]{1,2})", text, re.IGNORECASE)
        if due_day_m:
            entities["emi_day"] = int(due_day_m.group(1))

        # 9. Start Date
        start_d = self._extract_date(text, [
            r"(?:start\s*date|disbursement\s*date|sanction\s*date)[\s:]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ])
        if start_d:
            entities["start_date"] = start_d

        return entities

    def _extract_warranty_entities(self, text: str) -> Dict[str, Any]:
        """Extracts product warranty entities."""
        entities: Dict[str, Any] = {}

        # 1. Product Name / Model
        prod_m = re.search(r"(?:product|item|model|appliance)[\s:]*([A-Za-z0-9\s\-\.\+]{3,50})", text, re.IGNORECASE)
        if prod_m:
            entities["product_name"] = prod_m.group(1).strip()

        # 2. Serial Number
        serial_m = re.search(r"(?:serial\s*(?:no|number|num)|s\/n)[\s:]*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
        if serial_m:
            entities["serial_number"] = serial_m.group(1).strip()

        # 3. Purchase Date
        purchase_d = self._extract_date(text, [
            r"(?:purchase\s*date|date\s*of\s*purchase|invoice\s*date)[\s:]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ])
        if purchase_d:
            entities["purchase_date"] = purchase_d

        # 4. Warranty Expiry Date
        expiry_d = self._extract_date(text, [
            r"(?:warranty\s*valid\s*until|warranty\s*expiry|expires\s*on|valid\s*thru)[\s:]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ])
        if expiry_d:
            entities["expiry_date"] = expiry_d

        # 5. Warranty Duration Months
        dur_m = re.search(r"(?:warranty\s*period|duration)[\s:]*([0-9]+)\s*(?:months|yrs|years)?", text, re.IGNORECASE)
        if dur_m:
            val = int(dur_m.group(1))
            entities["duration_months"] = val * 12 if "year" in dur_m.group(0).lower() else val

        return entities

    def _extract_bill_entities(self, text: str) -> Dict[str, Any]:
        """Extracts utility bill entities."""
        entities: Dict[str, Any] = {}

        # 1. Provider
        for provider in self.KNOWN_UTILITY_PROVIDERS:
            if re.search(r"\b" + re.escape(provider) + r"\b", text, re.IGNORECASE):
                entities["provider"] = provider
                break

        # 2. Bill Type
        lower = text.lower()
        if "electricity" in lower or "bescom" in lower or "power" in lower:
            entities["bill_type"] = "Electricity"
        elif "water" in lower or "bwssb" in lower:
            entities["bill_type"] = "Water"
        elif "broadband" in lower or "airtel" in lower or "fiber" in lower:
            entities["bill_type"] = "Broadband"
        elif "gas" in lower or "igl" in lower or "mgl" in lower:
            entities["bill_type"] = "Gas"
        else:
            entities["bill_type"] = "Utility"

        # 3. Consumer / Account Number
        cons_m = re.search(r"(?:consumer\s*(?:no|number|id)|account\s*number|ca\s*no)[\s:]*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
        if cons_m:
            entities["consumer_number"] = cons_m.group(1).strip()

        # 4. Total Amount Due
        amount = self._extract_amount(text, [
            r"(?:total\s*amount\s*due|amount\s*payable|net\s*amount|bill\s*amount)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?:amount)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if amount:
            entities["amount_due"] = amount

        # 5. Due Date
        due_d = self._extract_date(text, [
            r"(?:due\s*date|pay\s*by|payment\s*due\s*date)[\s:]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ])
        if due_d:
            entities["due_date"] = due_d

        return entities

    def _extract_salary_entities(self, text: str) -> Dict[str, Any]:
        """Extracts salary payslip entities."""
        entities: Dict[str, Any] = {}
        net_pay = self._extract_amount(text, [
            r"(?:net\s*pay|net\s*salary|take\s*home)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if net_pay:
            entities["net_pay"] = net_pay
        gross_pay = self._extract_amount(text, [
            r"(?:gross\s*pay|gross\s*earnings|total\s*earnings)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if gross_pay:
            entities["gross_pay"] = gross_pay
        return entities

    def _extract_generic_entities(self, text: str) -> Dict[str, Any]:
        """Generic fallback for unrecognized document types."""
        entities: Dict[str, Any] = {}
        amt = self._extract_amount(text, [
            r"(?:total|amount|grand\s*total)[\s:]*(?:rs\.?|₹|inr)?\s*([0-9,]+(?:\.[0-9]{2})?)",
        ])
        if amt:
            entities["amount"] = amt
        d = self._extract_date(text, [
            r"(?:date)[\s:]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ])
        if d:
            entities["date"] = d
        return entities

    async def sync_entities_to_system(
        self, db: AsyncSession, user_id: uuid.UUID, doc: Document
    ) -> Tuple[Dict[str, Any], List[Event]]:
        """
        Synchronizes extracted document entities with system tables:
        - Insurance -> insurance_policies table + Event (INSURANCE_RENEWAL)
        - Loan -> loans table + Event (LOAN_EMI)
        - Warranty -> assets table + Event (WARRANTY_EXPIRY)
        - Bill -> Event (BILL_DUE)
        """
        entities = doc.extracted_entities or {}
        synced_records: Dict[str, Any] = {}
        generated_events: List[Event] = []

        if doc.document_type == DocumentType.INSURANCE:
            provider = entities.get("provider", "Unknown Insurer")
            pol_num = entities.get("policy_number")
            premium = entities.get("premium_amount")
            sum_ins = entities.get("sum_insured")
            expiry = entities.get("expiry_date")
            ins_type_str = entities.get("insurance_type", "health").upper()
            ins_type = getattr(InsuranceType, ins_type_str, InsuranceType.HEALTH)

            expiry_dt = date.fromisoformat(expiry) if expiry else None

            # Create or update Insurance policy
            policy = Insurance(
                user_id=user_id,
                name=f"{provider} {ins_type.value.capitalize()} Policy",
                insurance_type=ins_type,
                provider=provider,
                policy_number=pol_num,
                premium_amount=Decimal(str(premium)) if premium else None,
                sum_insured=Decimal(str(sum_ins)) if sum_ins else None,
                expiry_date=expiry_dt,
                next_premium_date=expiry_dt,
                metadata_json={"document_id": str(doc.id)},
            )
            db.add(policy)
            await db.flush()
            synced_records["insurance_id"] = str(policy.id)

            # Generate Renewal Event
            if expiry_dt:
                event = Event(
                    user_id=user_id,
                    event_type=EventType.INSURANCE_RENEWAL,
                    title=f"Insurance Renewal: {provider}",
                    description=f"Renewal due for policy {pol_num or ''}. Premium: ₹{premium or 0:,.2f}",
                    event_date=expiry_dt,
                    amount=Decimal(str(premium)) if premium else None,
                    status=EventStatus.UPCOMING,
                    confidence=0.95,
                    source_type="insurance",
                    source_id=policy.id,
                )
                db.add(event)
                generated_events.append(event)

        elif doc.document_type == DocumentType.LOAN:
            lender = entities.get("lender", "Bank")
            loan_type_str = entities.get("loan_type", "personal").upper()
            loan_type = getattr(LoanType, loan_type_str, LoanType.PERSONAL)
            principal = entities.get("principal_amount", 100000)
            emi = entities.get("emi_amount", 5000)
            roi = entities.get("interest_rate", 10.0)
            tenure = entities.get("tenure_months", 36)
            emi_day = entities.get("emi_day", 10)
            start_d_str = entities.get("start_date")

            start_d = date.fromisoformat(start_d_str) if start_d_str else date.today()
            end_d = start_d + timedelta(days=tenure * 30)

            loan = Loan(
                user_id=user_id,
                name=f"{lender} {loan_type.value.capitalize()} Loan",
                loan_type=loan_type,
                lender=lender,
                account_number_masked=entities.get("account_number", "")[-4:],
                principal_amount=Decimal(str(principal)),
                outstanding_amount=Decimal(str(principal)),
                interest_rate=roi,
                emi_amount=Decimal(str(emi)),
                start_date=start_d,
                end_date=end_d,
                emi_day=emi_day,
                total_emis=tenure,
                status=LoanStatus.ACTIVE,
                metadata_json={"document_id": str(doc.id)},
            )
            db.add(loan)
            await db.flush()
            synced_records["loan_id"] = str(loan.id)

            # Generate upcoming EMI Event (next occurrence)
            today = date.today()
            # Calculate next emi date
            next_month = today.month + 1 if today.day > emi_day else today.month
            next_year = today.year + (1 if next_month > 12 else 0)
            next_month = 1 if next_month > 12 else next_month
            next_emi_date = date(next_year, next_month, min(emi_day, 28))

            event = Event(
                user_id=user_id,
                event_type=EventType.LOAN_EMI,
                title=f"Loan EMI: {lender}",
                description=f"Monthly EMI for {loan.name}",
                event_date=next_emi_date,
                amount=Decimal(str(emi)),
                status=EventStatus.UPCOMING,
                confidence=1.0,
                source_type="loan",
                source_id=loan.id,
                is_recurring=True,
                recurrence_pattern={"frequency": "monthly", "day": emi_day},
            )
            db.add(event)
            generated_events.append(event)

        elif doc.document_type == DocumentType.WARRANTY:
            prod_name = entities.get("product_name", "Electronic Appliance")
            serial_no = entities.get("serial_number")
            purchase_d = entities.get("purchase_date")
            expiry_d = entities.get("expiry_date")

            exp_dt = date.fromisoformat(expiry_d) if expiry_d else None
            purch_dt = date.fromisoformat(purchase_d) if purchase_d else None

            asset = Asset(
                user_id=user_id,
                name=prod_name,
                asset_type=AssetType.ELECTRONICS,
                serial_number=serial_no,
                purchase_date=purch_dt,
                warranty_expiry=exp_dt,
                metadata_json={"document_id": str(doc.id)},
            )
            db.add(asset)
            await db.flush()
            synced_records["asset_id"] = str(asset.id)

            if exp_dt:
                event = Event(
                    user_id=user_id,
                    event_type=EventType.WARRANTY_EXPIRY,
                    title=f"Warranty Expiry: {prod_name}",
                    description=f"Manufacturer warranty for {prod_name} expires",
                    event_date=exp_dt,
                    status=EventStatus.UPCOMING,
                    confidence=0.90,
                    source_type="asset",
                    source_id=asset.id,
                )
                db.add(event)
                generated_events.append(event)

        elif doc.document_type == DocumentType.BILL:
            provider = entities.get("provider", "Utility Provider")
            amount = entities.get("amount_due")
            due_d = entities.get("due_date")
            due_dt = date.fromisoformat(due_d) if due_d else None

            if due_dt:
                event = Event(
                    user_id=user_id,
                    event_type=EventType.BILL_DUE,
                    title=f"Bill Due: {provider}",
                    description=f"{entities.get('bill_type', 'Utility')} bill due for payment",
                    event_date=due_dt,
                    amount=Decimal(str(amount)) if amount else None,
                    status=EventStatus.UPCOMING,
                    confidence=0.95,
                    source_type="document",
                    source_id=doc.id,
                )
                db.add(event)
                generated_events.append(event)

        # Audit log for document processing
        audit = AuditLog(
            user_id=user_id,
            action="document_processed",
            entity_type="document",
            entity_id=str(doc.id),
            details={
                "document_type": doc.document_type,
                "synced_records": synced_records,
                "events_created": len(generated_events),
            },
        )
        db.add(audit)
        await db.commit()

        return synced_records, generated_events
