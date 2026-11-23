# LIFEOS — Personal Life Operating System

> **A stateful AI decision engine for Indian households that continuously models financial and household state, detects subtle anomalies, forecasts upcoming contractual obligations, stress-tests against macroeconomic shocks, and mathematically simulates the real-world consequences of everyday purchasing decisions.**

---

## 🏛️ System Architecture

```
Data Ingestion → Life State Engine → Detection & RAG → Forecasting → Decision Simulator → Grounded Agent
     │                     │                │              │                  │                │
(Bank CSV/PDF)      (Runway & DTI)    (Spikes & Policy) (Floor+Quantile)   (Cash vs EMI)     (Zero Hallucination)
```

LIFEOS is **not a chatbot with a dashboard**. It is an explainable, deterministic intelligence pipeline built specifically for Indian financial realities (UPI, EMIs, NACH auto-debits, FD/RD, gold loans, tax timelines, and medical inflation):

| Intelligence Layer | Mechanism & Implementation |
|---|---|
| **1. Transaction Intelligence** | Hybrid TF-IDF Vectorizer (ngram 1-3) + Logistic Regression (15 classes, 94.48% accuracy, 0.9611 F1) |
| **2. Recurring Payment Engine** | Alias normalization, cadence clustering (7d, 30d, 90d, 365d), and amount variance analysis |
| **3. Anomaly & Fraud Detection** | Rolling IQR fences ($Q_{75} + 2.5 \times IQR$), 48-hour duplicate charge traps, salary drop alerts, and Isolation Forest |
| **4. Document Intelligence** | Native PDF & OCR entity extraction (Policy numbers, sum insured, loan tenures, warranty dates) via `pypdf` |
| **5. Future Event Engine** | Contractual obligation roll-forward, overdue status tracking, and 30/60/90-day horizon cashflow forecasting |
| **6. Deterministic Life State** | Mathematical state snapshots: liquid balance, emergency runway in months, savings rate, and composite health |
| **7. Expense Forecasting** | Contractual baseline floor + Indian seasonal demand weighting (Summer cooling, Diwali, festive travel) + P10/P50/P90 quantiles |
| **8. 5-Pillar Risk Engine** | Liquidity, DTI, Cashflow Margin, Obligation Horizon, and Expense Concentration scoring (Grades A through D) |
| **9. Macro Stress Testing** | 5 automated shocks: Complete 3-Month Income Loss, 6-Month Sabbatical, +200 bps Interest Spike, ₹1L Medical Emergency, +15% Inflation |
| **10. Goal & Milestone Engine** | Target amount accumulation, priority queuing, and automatic completion triggers |
| **11. Decision Simulator** | Multi-path purchase scenario optimization ("Can I afford X?"): Pay Cash vs 12M EMI vs Save Up & Delay |
| **12. Offline Document RAG** | ChromaDB local vector store with 2048-dim deterministic local embeddings (0 network dependency, 0 API costs) |
| **13. Grounded Agentic AI** | 13 typed, read-only deterministic tools orchestrating the core engine with 0% numerical hallucination |

---

## ⚡ Tech Stack

- **Backend**: Python 3.10+, FastAPI 0.115, Pydantic v2, SQLAlchemy 2.0 (Async)
- **Databases**:
  - SQLite with `aiosqlite` (zero-config local development and testing)
  - PostgreSQL / MySQL ready (via asyncpg / asyncmy)
  - ChromaDB (local persistent vector store for offline RAG)
- **Machine Learning**: scikit-learn, XGBoost, NumPy, Pandas, Joblib
- **Testing**: Pytest, Pytest-Asyncio, HTTPX (81 automated tests, 100% pass rate)

---

## 🚀 Quick Start (Running Locally)

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Git

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/ayush/LIFEOS.git
cd LIFEOS/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install all verified dependencies
pip install -r requirements.txt
```

### 3. Run the Live Guide Demonstration (~3 seconds)

A self-contained showcase script demonstrating all 13 core intelligence layers end-to-end with zero mock network dependencies:

```bash
python demo_guide.py
```

### 4. Run the Full Test Suite (81 Tests)

```bash
pytest tests/ -v
```

All 81 tests execute against an isolated in-memory SQLite database in under 25 seconds:
- `test_agent.py` — Agent tool execution, read-only safety guardrails, intent routing
- `test_anomaly_detection.py` — Duplicate traps, utility spikes, ground truth evaluation
- `test_auth.py` — JWT authentication, registration, password hashing
- `test_document_intelligence.py` — PDF parsing, insurance/loan entity synchronization
- `test_event_engine.py` — Horizon calculations, recurring rollforward
- `test_forecast_engine.py` — Contractual floor, seasonal adjustments, MAPE backtesting
- `test_goals.py` — Financial goals CRUD & automatic completion
- `test_health.py` — Service health & category auto-seeding
- `test_isolation.py` — Strict multi-tenant user data isolation
- `test_life_state_engine.py` — Runway, net worth, state deltas
- `test_rag_service.py` — ChromaDB semantic indexing, chunking, citation retrieval
- `test_recurring_detection.py` — Merchant normalization, cadence clustering
- `test_risk_engine.py` — 5-pillar composite scoring, stress test execution
- `test_scenario_simulator.py` — Purchase simulator, tenure comparison
- `test_synthetic_data.py` — Indian household data generation & reproducibility
- `test_transaction_intelligence.py` — ML model predictions, CSV imports

### 5. Launch the FastAPI Server & Interactive API Docs

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser to explore the **58 OpenAPI endpoints** via Swagger UI:
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📂 Project Structure

```
LIFEOS/
├── backend/
│   ├── app/
│   │   ├── api/                # 17 FastAPI routers (58 endpoints)
│   │   │   ├── agent.py        # Conversational agent & tool introspection
│   │   │   ├── alerts.py       # Anomaly alerts & lifecycle
│   │   │   ├── assets.py       # Physical/digital assets & TCO
│   │   │   ├── auth.py         # JWT register & login
│   │   │   ├── dashboard.py    # Overview, "What Changed?", "What Am I Forgetting?"
│   │   │   ├── documents.py    # PDF upload & entity extraction
│   │   │   ├── events.py       # Future obligations & horizons
│   │   │   ├── forecast.py     # Expense forecasting & backtesting
│   │   │   ├── goals.py        # Goal tracking & milestones
│   │   │   ├── lifestate.py    # Life state computation & history
│   │   │   ├── loans.py        # Active loans & EMI tracking
│   │   │   ├── rag.py          # Document RAG semantic search
│   │   │   ├── recurring.py    # Cadence detection & confirmations
│   │   │   ├── risk.py         # 5-Pillar risk scoring & stress tests
│   │   │   ├── simulations.py  # Purchase decision simulator
│   │   │   ├── subscriptions.py# Recurring digital subscriptions
│   │   │   └── transactions.py # ML categorization & CSV imports
│   │   ├── database/           # Async SQLAlchemy session & table init
│   │   ├── ml/                 # TF-IDF + Logistic Regression pipeline
│   │   ├── models/             # 17 declarative SQLAlchemy models
│   │   ├── schemas/            # Pydantic v2 validation schemas
│   │   ├── services/           # 11 core intelligence engines
│   │   ├── config.py           # Centralized settings & environment
│   │   └── main.py             # FastAPI entrypoint & category seeder
│   ├── ml_models/              # Trained ML model artifact (.joblib)
│   ├── tests/                  # 16 test files (81 tests, 100% pass)
│   ├── demo_guide.py           # Self-contained guide demonstration script
│   └── requirements.txt        # Verified production dependencies
├── data/
│   └── synthetic/              # 29,731 Indian household transactions & ground truth
├── docs/
│   ├── ARCHITECTURE.md         # Comprehensive architectural blueprint
│   ├── PROGRESS.md             # Phase-by-phase completion record
│   └── GUIDE_PRESENTATION_DOSSIER.md # Academic viva & evaluation guide
├── .gitignore
└── README.md
```

---

## 🎓 Academic Presentation & Guide Viva Highlights

When presenting LIFEOS to an academic committee or project guide:

1. **Why not just use an LLM API directly?**
   - General-purpose LLMs hallucinate financial balances and calculations. LIFEOS uses a **dual-layer architecture**: all math, risk ratios (DTI, runway), and stress tests are calculated **deterministically in Python**. The LLM layer acts strictly as a tool caller and natural language interface with **0% numerical hallucination**.
2. **How is the Indian household context modeled?**
   - Supports UPI merchant formats (`UPI/swiggy/ref-9812`), NACH auto-debits, festival seasonality (Diwali, Summer school fees), joint household commitments, and high DTI stress thresholds typical of tier-1/tier-2 Indian families.
3. **Privacy & Offline Operation:**
   - Vector indexing and semantic retrieval are powered by local ChromaDB and zero-network embeddings. All customer financial data remains strictly local or on-premise.

---

## 📄 License

Academic / Portfolio Project. All rights reserved.
