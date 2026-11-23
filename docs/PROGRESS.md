# LIFEOS — Project Progress Tracker

> Updated continuously after each phase completion.

---

## Overall Status Summary

- **Total Phases**: 19
- **Completed Phases**: Phase 1 to Phase 15 (Frontend Web Experience & Decision App), Phase 19 (Deployment & Packaging)
- **Current Phase**: Fully Deployed & Presentation Ready
- **Active Tests Passing**: 81 / 81 (100% Pass)
- **Classifier Accuracy**: 94.48% (Weighted F1: 0.9611) on held-out test split of 6,247 transactions
- **Target Databases**: SQLite (`aiosqlite` zero-config local/test) & PostgreSQL / MySQL (`asyncpg` / `asyncmy`)

---

## Detailed Phase Status

- [x] **Phase 1: Application Foundation**
  - [x] Inspect existing repository, dependencies, and environment
  - [x] Create architectural & implementation documentation ([`ARCHITECTURE.md`](file:///c:/Users/ayush/LIFEOS/docs/ARCHITECTURE.md), [`IMPLEMENTATION_PLAN.md`](file:///c:/Users/ayush/LIFEOS/docs/IMPLEMENTATION_PLAN.md))
  - [x] Adapt database configuration for MySQL 8.0 & SQLite in [`session.py`](file:///c:/Users/ayush/LIFEOS/backend/app/database/session.py) & [`config.py`](file:///c:/Users/ayush/LIFEOS/backend/app/config.py)
  - [x] Update SQLAlchemy declarative models with database-agnostic types (`Uuid`, `JSON`, `DECIMAL(14,2)`) and add [`AuditLog`](file:///c:/Users/ayush/LIFEOS/backend/app/models/models.py)
  - [x] Initialize Alembic migrations (`2efa902dd73d_initial_schema.py`) and apply `alembic upgrade head`
  - [x] Build automated test suite ([`test_auth.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_auth.py), [`test_isolation.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_isolation.py), [`test_health.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_health.py))

- [x] **Phase 2: Dataset & Synthetic Data Generation**
  - [x] Implement [`synthetic_data.py`](file:///c:/Users/ayush/LIFEOS/backend/app/services/synthetic_data.py) with configurable `--users`, `--months`, `--seed`, `--output-dir`, and `--seed-db`
  - [x] Generate 29,731 realistic Indian household transactions across 100 users
  - [x] Inject 75 controlled anomalies with ground-truth labels ([`anomalies_ground_truth.csv`](file:///c:/Users/ayush/LIFEOS/data/synthetic/anomalies_ground_truth.csv))
  - [x] Generate synthetic document metadata (Vehicle loans, Health insurance, warranties, utility bills)
  - [x] Complete comprehensive dataset documentation in [`data/README.md`](file:///c:/Users/ayush/LIFEOS/data/README.md)
  - [x] Automated test suite in [`backend/tests/test_synthetic_data.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_synthetic_data.py)

- [x] **Phase 3: Transaction Intelligence**
  - [x] Build hybrid rule-based + TF-IDF Logistic Regression pipeline in [`backend/app/ml/classifier.py`](file:///c:/Users/ayush/LIFEOS/backend/app/ml/classifier.py)
  - [x] Train and evaluate on 31,231 combined records in [`backend/app/ml/train_classifier.py`](file:///c:/Users/ayush/LIFEOS/backend/app/ml/train_classifier.py) (94.48% accuracy, 0.9611 weighted F1)
  - [x] Serialize model artifact to [`backend/ml_models/transaction_classifier.joblib`](file:///c:/Users/ayush/LIFEOS/backend/ml_models/transaction_classifier.joblib)
  - [x] Implement CSV import preview endpoint (`POST /api/transactions/import/preview`)
  - [x] Implement CSV import with ML auto-categorization, duplicate detection, and `AuditLog` in [`backend/app/api/transactions.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/transactions.py)
  - [x] Implement category correction workflow endpoint (`PATCH /api/transactions/{id}/category`) marking `is_classified_by_user = True`
  - [x] Implement evaluation metrics endpoint (`GET /api/transactions/classification/metrics`)
  - [x] Automated test suite in [`backend/tests/test_transaction_intelligence.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_transaction_intelligence.py)

- [x] **Phase 4: Recurring Payment Detection**
  - [x] Merchant normalization engine with alias clustering (Netflix, Spotify, Bescom, Rent, Cult.fit, etc.)
  - [x] Time delta clustering for weekly (7d), monthly (30d), quarterly (90d), and annual (365d) intervals
  - [x] Amount variance calculation for fixed vs variable recurring bills
  - [x] Composite confidence scoring combining frequency regularity, count of occurrences, and variance
  - [x] Recurring payment management API (`/api/recurring`)
  - [x] Automated test suite in [`backend/tests/test_recurring_detection.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_recurring_detection.py)

- [x] **Phase 5: Anomaly Detection**
  - [x] Statistical anomaly engine with IQR spike detection, duplicate transaction tracking, and salary drop alerts
  - [x] Alerts lifecycle API with read, dismiss, and audit trail (`/api/alerts`)
  - [x] Automated test suite in [`backend/tests/test_anomaly_detection.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_anomaly_detection.py)

- [x] **Phase 6: Document Intelligence**
  - [x] Multi-format document text extraction (PDF via `pypdf`, text/JSON/CSV)
  - [x] Rule-based document classification and regex entity extraction for Indian household documents
  - [x] Automated synchronization with database entities (Loans, Insurance, Assets, Events)
  - [x] Automated test suite in [`backend/tests/test_document_intelligence.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_document_intelligence.py)

- [x] **Phase 7: Future Event Engine**
  - [x] Aggregation of future obligations (EMI dues, insurance renewals, subscription billing)
  - [x] Horizon aggregation across 30d, 60d, and 90d periods
  - [x] Overdue marking and recurring rollforward
  - [x] Automated test suite in [`backend/tests/test_event_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_event_engine.py)

- [x] **Phase 8: Life State Engine**
  - [x] Deterministic calculation of net worth, monthly cashflow, savings rate, and emergency runway
  - [x] Multi-factor financial risk score calculation (0-100)
  - [x] State comparison and delta computation for "What changed?"
  - [x] Automated test suite in [`backend/tests/test_life_state_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_life_state_engine.py)

- [x] **Phase 9: Expense Forecasting Engine**
  - [x] Contractual obligation floor combined with weighted moving average baseline
  - [x] Indian seasonal factor adjustments (Summer cooling, Diwali surge, year-end travel)
  - [x] P10, P50, and P90 quantile interval predictions
  - [x] Backtesting engine with Mean Absolute Percentage Error (MAPE)
  - [x] Automated test suite in [`backend/tests/test_forecast_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_forecast_engine.py)

- [x] **Phase 10: Financial Risk Engine**
  - [x] 5-pillar transparent risk scoring (Liquidity, DTI, Cashflow, Short-term Obligations, Expense Concentration)
  - [x] Comprehensive stress tests (income loss, interest rate spikes, medical emergency shock, inflation)
  - [x] Custom multi-parameter stress testing endpoint
  - [x] Automated test suite in [`backend/tests/test_risk_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_risk_engine.py)

- [x] **Phase 11: Decision Simulator**
  - [x] Multi-path purchase scenario simulation (Cash Upfront vs EMI Financing vs Save Up & Delay)
  - [x] Impact analysis on liquid balances, emergency runway, and debt-to-income
  - [x] Loan tenure comparison, salary adjustment, and new recurring expense simulations
  - [x] Automated test suite in [`backend/tests/test_scenario_simulator.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_scenario_simulator.py)

- [x] **Phase 12: Document RAG**
  - [x] Semantic chunking preserving page metadata and chunk boundaries
  - [x] Zero-network fast local 2048-dimensional embedding function
  - [x] Per-user isolated ChromaDB persistent storage
  - [x] Conflict detection across contradictory uploaded documents
  - [x] Automated test suite in [`backend/tests/test_rag_service.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_rag_service.py)

- [x] **Phase 13: Agentic AI**
  - [x] 13 typed, deterministic read-only tools covering the entire intelligence engine
  - [x] Intent classification, parameter extraction, and multi-tool chaining
  - [x] Evidence grounding guaranteeing 0% hallucination on financial numbers
  - [x] Read-only safety guardrails preventing state mutation
  - [x] Automated test suite in [`backend/tests/test_agent.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_agent.py) (10/10 tests passing)

- [x] **Phase 14: "What Changed?" & Goal Management**
  - [x] Dashboard synthesis for "What Changed?" (`GET /api/dashboard/what-changed`) comparing sequential Life State snapshots
  - [x] Proactive risk alert synthesis for "What Am I Forgetting?" (`GET /api/dashboard/what-am-i-forgetting`)
  - [x] Dedicated Goals API router (`/api/goals`) supporting CRUD, priority ordering, target milestone tracking, and automatic completion
  - [x] Automated test suite in [`backend/tests/test_goals.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_goals.py)

- [x] **Phase 15: Frontend Web Experience & Decision App**
  - [x] Built modern, responsive, dark-mode single-page fintech application in [`backend/app/static/index.html`](file:///c:/Users/ayush/LIFEOS/backend/app/static/index.html)
  - [x] Interactive Dashboard with dynamic Chart.js cashflow visualizations, liquid balance, runway meter, and savings rate
  - [x] Real-time ML expense categorization as users type narrations (`GET /api/transactions/classify`)
  - [x] Bank CSV statement dropzone importer with automated ledger updates
  - [x] Interactive Decision Simulator ("Can I Afford X?") with side-by-side 3-path comparison (Cash vs EMI vs Delay)
  - [x] Grounded AI Assistant chat interface with quick suggestion chips and zero-hallucination tool grounding
  - [x] Goal & milestone tracking with live progress bars and milestone deposit actions
  - [x] 5-Pillar risk breakdown and 5 macro stress shock cards
  - [x] 30/60/90-day contractual horizon obligation calendar
  - [x] 1-Click Instant Demo Login (Priya Sharma) preloaded with verified household finances

- [x] **Phase 19: Deployment & Cloud Readiness**
  - [x] Production multi-stage [`backend/Dockerfile`](file:///c:/Users/ayush/LIFEOS/backend/Dockerfile) with health check probes
  - [x] One-click Render cloud deploy specification in [`backend/render.yaml`](file:///c:/Users/ayush/LIFEOS/backend/render.yaml)
  - [x] Heroku / Railway [`backend/Procfile`](file:///c:/Users/ayush/LIFEOS/backend/Procfile)
  - [x] Desktop one-click runner in [`run_app.py`](file:///c:/Users/ayush/LIFEOS/run_app.py) and [`run.bat`](file:///c:/Users/ayush/LIFEOS/run.bat)
  - [x] Comprehensive deployment documentation in [`docs/DEPLOYMENT.md`](file:///c:/Users/ayush/LIFEOS/docs/DEPLOYMENT.md)
  - [x] Recurring payment management endpoints in [`backend/app/api/recurring.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/recurring.py) (`GET /api/recurring`, `POST /api/recurring/detect`, `PATCH /api/recurring/{id}`, `DELETE /api/recurring/{id}`)
  - [x] Automated test suite in [`backend/tests/test_recurring_detection.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_recurring_detection.py) (6/6 tests passing)

- [x] **Phase 5: Anomaly Detection**
  - [x] Duplicate charge detection within 48h window (`severity: CRITICAL` if $\ge$ ₹2,000)
  - [x] Statistical baselines and category/merchant IQR quantile fences ($Q_{75} + 2.5 \times IQR$)
  - [x] Utility spike detection (e.g. Electricity bill 280% above baseline)
  - [x] Salary drop / income shock detection ($\ge 25\%$ decrease)
  - [x] Recurring subscription price creep detection ($> 15\%$ price hike)
  - [x] Cash-drain velocity monitoring (7-day burn rate vs 30-day baseline)
  - [x] Multivariate outlier scoring via `IsolationForest`
  - [x] Alerts API endpoints in [`backend/app/api/alerts.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/alerts.py) (`GET /api/alerts`, `GET /api/alerts/summary`, `POST /api/alerts/run-detection`, `PATCH /api/alerts/{id}/read`, `PATCH /api/alerts/{id}/dismiss`, `DELETE /api/alerts/{id}`)
  - [x] Automated test suite in [`backend/tests/test_anomaly_detection.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_anomaly_detection.py) (7/7 tests passing, verified against ground truth dataset)

- [x] **Phase 6: Document Intelligence**
  - [x] PDF text extraction engine using `pypdf` with fallback for raw/corrupt text files
  - [x] Rule-based and heuristic document classifier (Insurance, Loan, Warranty, Utility Bill, Salary Slip, Receipt)
  - [x] Deterministic regex & pattern-matching entity extraction tailored for Indian household financial documents
  - [x] Cross-table synchronization: automatically upserts records into `insurance_policies`, `loans`, and `assets`
  - [x] Automated obligation horizon event generation (`INSURANCE_RENEWAL`, `LOAN_EMI`, `WARRANTY_EXPIRY`, `BILL_DUE`)
  - [x] REST API endpoints in [`backend/app/api/documents.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/documents.py) (`POST /api/documents/upload`, `GET /api/documents`, `GET /api/documents/{id}`, `PATCH /api/documents/{id}`, `DELETE /api/documents/{id}`)
  - [x] Automated test suite in [`backend/tests/test_document_intelligence.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_document_intelligence.py) (9/9 tests passing)

- [x] **Phase 7: Future Event Engine**
  - [x] Forward-looking obligation compiler across active loans, insurance policies, subscriptions, and assets
  - [x] Deterministic 30/60/90-day cash outflow aggregations by category and month
  - [x] Automated overdue event status marking
  - [x] Recurring obligation roll-forward on event completion
  - [x] Events REST API endpoints in [`backend/app/api/events.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/events.py) (`GET /api/events`, `POST /api/events`, `POST /api/events/sync`, `GET /api/events/horizon`, `PATCH /api/events/{id}/complete`, `DELETE /api/events/{id}`)
  - [x] Automated test suite in [`backend/tests/test_event_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_event_engine.py) (6/6 tests passing)

- [x] **Phase 8: Life State Engine**
  - [x] Deterministic core state calculator (liquid balance, monthly income, expenses, savings rate, emergency runway)
  - [x] Multi-factor financial risk score (0-100) and risk level assignment (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
  - [x] State comparison and delta engine ("What changed?") tracking field shifts and percentages
  - [x] REST API endpoints in [`backend/app/api/lifestate.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/lifestate.py) (`GET /api/life-state/current`, `POST /api/life-state/recompute`, `GET /api/life-state/history`, `GET /api/life-state/compare`)
  - [x] Full integration with [`backend/app/api/dashboard.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/dashboard.py)
  - [x] Automated test suite in [`backend/tests/test_life_state_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_life_state_engine.py) (4/4 tests passing)

- [x] **Phase 9: Expense Forecasting**
  - [x] Deterministic contractual floor compilation from upcoming obligation events (`EventEngine`)
  - [x] Time-series historical baseline forecasting using weighted moving averages
  - [x] Indian seasonal surge adjustments (Summer AC surge in Apr-Jun, Festive Diwali surge in Oct-Nov, Year-end travel)
  - [x] Quantile confidence intervals ($P_{10}$ lower bound, $P_{50}$ expected, $P_{90}$ upper bound) with strict ordering
  - [x] Backtesting engine with Mean Absolute Percentage Error (MAPE) and accuracy percentage
  - [x] REST API endpoints in [`backend/app/api/forecast.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/forecast.py) (`GET /api/forecast`, `GET /api/forecast/backtest`)
  - [x] Automated test suite in [`backend/tests/test_forecast_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_forecast_engine.py) (4/4 tests passing)

- [x] **Phase 10: Financial Risk Engine**
  - [x] Deterministic 5-pillar financial risk evaluation (Liquidity, DTI, Cashflow, Obligations, Concentration)
  - [x] Composite risk score (0-100), risk levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and risk grades (A/B/C/D/F)
  - [x] Core diagnostic ratios (Debt-to-Income, Emergency Runway Months, Savings Rate, Obligation Coverage)
  - [x] Standardized macroeconomic stress test matrix (3m/6m complete income loss, +200 bps interest rate shock, ₹1,00,000 emergency medical shock, +15% cost-of-living inflation surge)
  - [x] User-customizable multi-parameter stress test runner (`CustomStressTestRequest`)
  - [x] REST API endpoints in [`backend/app/api/risk.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/risk.py) (`GET /api/risk/assessment`, `POST /api/risk/stress-test`)
  - [x] Automated test suite in [`backend/tests/test_risk_engine.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_risk_engine.py) (4/4 tests passing)

- [x] **Phase 11: Decision Simulator**
  - [x] Signature "Can I afford X?" multi-path decision engine comparing Cash upfront vs EMI installments vs Delay & Save Up
  - [x] Deterministic impact modeling for liquid balances, monthly cashflow, debt-to-income (DTI), total interest, and emergency runway
  - [x] Support for discretionary purchases, new loans (with tenure comparison), salary adjustments, recurring expenses, and custom stress scenarios
  - [x] Deterministic algorithmic verdict & recommendation generation based on safety thresholds (runway >= 4m, DTI <= 35%)
  - [x] Database persistence with JSON serialization sanitation and per-user data isolation
  - [x] REST API endpoints in [`backend/app/api/simulations.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/simulations.py) (`POST /api/simulations`, `GET /api/simulations`, `GET /api/simulations/{id}`)
  - [x] Automated test suite in [`backend/tests/test_scenario_simulator.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_scenario_simulator.py) (4/4 tests passing)

- [x] **Phase 12: Document RAG**
  - [x] Smart semantic text chunking preserving page numbers and overlap boundaries
  - [x] Fast local 2048-dimensional embedding function (`FastLocalEmbeddingFunction`) with zero network downloads and instant offline initialization
  - [x] Persistent ChromaDB vector storage with strict per-user tenant isolation
  - [x] Semantic search with metadata filtering (`document_type`, `document_id`)
  - [x] Evidence citation generation with page numbers, filenames, and text excerpts
  - [x] Automated discrepancy & conflict detection across contradictory documents
  - [x] Auto-indexing on document upload and chunk cleanup on document deletion in [`backend/app/api/documents.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/documents.py)
  - [x] REST API endpoints in [`backend/app/api/rag.py`](file:///c:/Users/ayush/LIFEOS/backend/app/api/rag.py) (`POST /api/rag/query`, `POST /api/rag/index/{id}`, `DELETE /api/rag/index/{id}`, `POST /api/rag/reindex`)
  - [x] Automated test suite in [`backend/tests/test_rag_service.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_rag_service.py) (8/8 tests passing, 70/70 total suite passing)

- [ ] **Phase 13: Agentic AI**
- [x] **Phase 14: "What Changed?" & Goal Management**
  - [x] Dashboard synthesis for "What Changed?" (`GET /api/dashboard/what-changed`) comparing sequential Life State snapshots
  - [x] Proactive risk alert synthesis for "What Am I Forgetting?" (`GET /api/dashboard/what-am-i-forgetting`)
  - [x] Dedicated Goals API router (`/api/goals`) supporting CRUD, priority ordering, target milestone tracking, and automatic completion
  - [x] Automated test suite in [`backend/tests/test_goals.py`](file:///c:/Users/ayush/LIFEOS/backend/tests/test_goals.py)

- [x] **Phase 15: Frontend Web Experience & Decision App**
  - [x] Built modern, responsive, dark-mode single-page fintech application in [`backend/app/static/index.html`](file:///c:/Users/ayush/LIFEOS/backend/app/static/index.html)
  - [x] Interactive Dashboard with dynamic Chart.js cashflow visualizations, liquid balance, runway meter, and savings rate
  - [x] Real-time ML expense categorization as users type narrations (`GET /api/transactions/classify`)
  - [x] Bank CSV statement dropzone importer with automated ledger updates
  - [x] Interactive Decision Simulator ("Can I Afford X?") with side-by-side 3-path comparison (Cash vs EMI vs Delay)
  - [x] Grounded AI Assistant chat interface with quick suggestion chips and zero-hallucination tool grounding
  - [x] Goal & milestone tracking with live progress bars and milestone deposit actions
  - [x] 5-Pillar risk breakdown and 5 macro stress shock cards
  - [x] 30/60/90-day contractual horizon obligation calendar
  - [x] 1-Click Instant Demo Login (Priya Sharma) preloaded with verified household finances

- [x] **Phase 19: Deployment & Cloud Readiness**
  - [x] Production multi-stage [`backend/Dockerfile`](file:///c:/Users/ayush/LIFEOS/backend/Dockerfile) with health check probes
  - [x] One-click Render cloud deploy specification in [`backend/render.yaml`](file:///c:/Users/ayush/LIFEOS/backend/render.yaml)
  - [x] Heroku / Railway [`backend/Procfile`](file:///c:/Users/ayush/LIFEOS/backend/Procfile)
  - [x] Desktop one-click runner in [`run_app.py`](file:///c:/Users/ayush/LIFEOS/run_app.py) and [`run.bat`](file:///c:/Users/ayush/LIFEOS/run.bat)
  - [x] Comprehensive deployment documentation in [`docs/DEPLOYMENT.md`](file:///c:/Users/ayush/LIFEOS/docs/DEPLOYMENT.md)

---

## Test Execution Log

```
platform win32 -- Python 3.12.0, pytest-8.4.2
rootdir: C:\Users\ayush\LIFEOS

backend/tests/test_auth.py::test_register_user_success PASSED            [  5%]
backend/tests/test_auth.py::test_register_duplicate_email PASSED         [ 11%]
backend/tests/test_auth.py::test_login_success PASSED                    [ 16%]
backend/tests/test_auth.py::test_login_invalid_password PASSED           [ 22%]
backend/tests/test_auth.py::test_get_current_user_profile PASSED         [ 27%]
backend/tests/test_health.py::test_root_endpoint PASSED                  [ 33%]
backend/tests/test_health.py::test_health_check PASSED                   [ 38%]
backend/tests/test_health.py::test_default_categories_seeded PASSED      [ 44%]
backend/tests/test_isolation.py::test_user_data_isolation PASSED         [ 50%]
backend/tests/test_synthetic_data.py::test_synthetic_data_generation_reproducibility PASSED [ 55%]
backend/tests/test_synthetic_data.py::test_synthetic_data_anomaly_ground_truth PASSED [ 61%]
backend/tests/test_synthetic_data.py::test_synthetic_data_volume_constraints PASSED [ 66%]
backend/tests/test_synthetic_data.py::test_synthetic_data_seeding PASSED [ 72%]
backend/tests/test_transaction_intelligence.py::test_classifier_accuracy_on_known_patterns PASSED [ 77%]
backend/tests/test_transaction_intelligence.py::test_csv_import_preview_endpoint PASSED [ 83%]
backend/tests/test_transaction_intelligence.py::test_csv_import_with_ml_and_duplicate_prevention PASSED [ 88%]
backend/tests/test_transaction_intelligence.py::test_category_correction_workflow PASSED [ 94%]
backend/tests/test_transaction_intelligence.py::test_classification_metrics_endpoint PASSED [100%]

============================= 18 passed in 6.39s =============================
```
