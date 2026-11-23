# LIFEOS — Master Implementation Plan

This document details the step-by-step implementation phases for building the **LIFEOS** personal finance and household intelligence platform.

---

## Phase Matrix

| Phase | Title | Core Objective | Key Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Application Foundation | Multi-DB support, SQLAlchemy models, Auth, Alembic, Base APIs | `models.py`, `session.py`, `config.py`, `auth.py`, test suite | 🟡 In Progress |
| **Phase 2** | Synthetic Data Generator | Reproducible Indian household financial dataset with ground truth | `synthetic_data.py`, CLI generator, seed scripts, `data/README.md` | ⚪ Scheduled |
| **Phase 3** | Transaction Intelligence | CSV/Excel import, column normalization, TF-IDF + Logistic Reg classifier | `classifier.py`, import pipeline, evaluation metrics, confidence threshold | ⚪ Scheduled |
| **Phase 4** | Recurring Payment Detection | Merchant normalization, interval clustering, next-due estimation | `recurring.py`, confidence scoring, user confirmation API | ⚪ Scheduled |
| **Phase 5** | Anomaly Detection | Rolling statistical baselines, quantile fences, Isolation Forest | `anomaly.py`, alert generation with evidence, user feedback loop | ⚪ Scheduled |
| **Phase 6** | Document Intelligence | Document upload, regex/OCR extraction, Pydantic schemas, provenance | `document_service.py`, field extraction, user verification API | ⚪ Scheduled |
| **Phase 7** | Future Event Horizon | Unified 30/60/90-day obligation engine from recurring, EMIs, docs | `event_engine.py`, timeline endpoints, status transition logic | ⚪ Scheduled |
| **Phase 8** | Life State Engine | Deterministic state snapshot, net surplus, emergency runway, diffs | `life_state.py`, state history, snapshot comparison service | ⚪ Scheduled |
| **Phase 9** | Expense Forecasting | Time-series forecasting (rolling average, seasonal naive, regression) | `forecasting.py`, MAE/RMSE evaluation, interval projection | ⚪ Scheduled |
| **Phase 10** | Financial Risk Engine | Explainable composite risk indicators, cashflow pressure, coverage | `risk_engine.py`, risk score calculation with explicit breakdown | ⚪ Scheduled |
| **Phase 11** | Decision Simulator | Deterministic purchase simulator (Cash vs EMI) & stress testing | `simulator.py`, buffer breach analysis, month-by-month trajectory | ⚪ Scheduled |
| **Phase 12** | Document RAG | Semantic chunking, embeddings, ChromaDB retrieval, user isolation | `rag_service.py`, vector indexing, citation generation | ⚪ Scheduled |
| **Phase 13** | Agentic AI & Tools | 13 typed tools, multi-step tool execution, grounded reasoning | `tools.py`, `agent_service.py`, read-only financial guardrails | ⚪ Scheduled |
| **Phase 14** | "What Changed?" & "What Am I Forgetting?" | First-class intelligence queries comparing snapshots & horizons | Specialized API endpoints and aggregation services | ⚪ Scheduled |
| **Phase 15** | Frontend Experience | Next.js + TypeScript + Tailwind + Recharts dashboard & views | Dashboard, Transactions, Timeline, Simulator, Docs, Agent UI | ⚪ Scheduled |
| **Phase 16** | API Design & Polish | OpenAPI specifications, standardized pagination, error handling | Clean Swagger docs, consistent error envelopes | ⚪ Scheduled |
| **Phase 17** | Security & Privacy | User isolation audit, input sanitization, safe error handling | Security tests, zero-PIN/OTP guarantees | ⚪ Scheduled |
| **Phase 18** | End-to-End Testing | Automated unit, integration, and ML evaluation test suites | `pytest` test suite, ML precision/recall benchmarks | ⚪ Scheduled |
| **Phase 19** | Deployment & Documentation | Dockerfiles, Docker Compose, setup guides, demo walk-throughs | `Dockerfile`, `docker-compose.yml`, seed guides, demo guide | ⚪ Scheduled |

---

## Detailed Acceptance Criteria per Phase

### Phase 1: Application Foundation
- [x] Multi-database support: Native MySQL 8.0 support (`asyncmy`) + zero-config local / test SQLite (`aiosqlite`).
- [ ] Complete declarative SQLAlchemy 2.0 schema for all 17 entities with `AuditLog` and Decimal arithmetic.
- [ ] User registration, bcrypt password hashing, JWT authentication with per-request user isolation.
- [ ] Alembic migration initialization.
- [ ] Automated test suite validating auth, table creation, and user data isolation.

### Phase 2: Synthetic Data Generation
- [ ] Configurable generator accepting random seeds (`--seed`, `--users`, `--months`).
- [ ] Minimum 10,000 transactions across realistic Indian household categories (groceries, utilities, fuel, UPI, EMIs).
- [ ] Controlled injected anomalies with ground-truth labels (spikes, unexpected merchants, duplicate debits).
- [ ] Synthetic document metadata generation (insurance, loans, utility bills, receipts).
- [ ] CLI command to populate database with demo data.

### Phase 3: Transaction Intelligence
- [ ] Robust CSV / statement importer supporting Indian bank formats (HDFC, SBI, ICICI, Axis, UPI).
- [ ] Machine learning classifier trained on personal finance dataset using TF-IDF + Classifier.
- [ ] Measured classification performance (Precision, Recall, F1 score).
- [ ] Low-confidence flag threshold (default < 0.70) routing to user review.

### Phase 4: Recurring Payment Detection
- [ ] Merchant name normalization (e.g., `NETFLIX MUMBAI`, `NETFLIX.COM` -> `Netflix`).
- [ ] Interval cadence detection (weekly, monthly, quarterly, annual).
- [ ] Expected amount range and next payment date projection.

### Phase 5: Anomaly Detection
- [ ] Rolling baseline comparison by category and merchant.
- [ ] Alert generation with evidence string, deviation percentage, and severity.
- [ ] User feedback action (Dismiss vs Mark as Expected).

### Phase 6: Document Intelligence
- [ ] Upload endpoint supporting PDF and images.
- [ ] Text extraction and structured Pydantic schema validation for insurance, loans, warranties, and utility bills.
- [ ] Automatic event creation from validated document expiration and due dates.

### Phase 7: Future Event Horizon
- [ ] Consolidated timeline for 30-, 60-, and 90-day horizons.
- [ ] Differentiation between confirmed dates, user reminders, and estimated recurring charges.

### Phase 8: Life State Engine
- [ ] Instantaneous snapshot computation: Liquid Cash, Monthly Income, Monthly Expense, Surplus, Runway, Debt Ratio.
- [ ] State history tracking and snapshot delta comparisons.

### Phase 9: Expense Forecasting
- [ ] Baseline forecasting comparing rolling average vs linear trend.
- [ ] Evaluation using chronological train/test splits (MAE, RMSE).

### Phase 10: Financial Risk Engine
- [ ] Explainable 0-100 composite risk score with explicit contributing factors.

### Phase 11: Decision Simulator
- [ ] Deterministic purchase simulation (Buy Now vs Delayed vs EMI).
- [ ] Stress-test simulation (income shock, emergency expense).

### Phase 12: Document RAG
- [ ] Document chunking and embedding storage in ChromaDB.
- [ ] User-scoped vector retrieval with source citations.

### Phase 13: Agentic AI
- [ ] 13 typed tools implemented and wired.
- [ ] Multi-tool execution loop with read-only safety guardrails.

### Phase 14: "What Changed?" & "What Am I Forgetting?"
- [ ] Dedicated endpoints providing transparent change detection and obligation auditing.

### Phase 15: Frontend Experience
- [ ] Next.js 14+ / TypeScript responsive interface with Tailwind CSS, Lucide icons, and Recharts.

### Phase 16 to 19: Polish, Security, Tests, and Deployment
- [ ] Full automated test suite passing across all modules.
- [ ] Complete setup and demo guide in README.
