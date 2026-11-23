# LIFEOS — Architecture Documentation

> **A Stateful AI Decision & Household Intelligence Engine**  
> *Data → State → Detection → Prediction → Simulation → Agent → Action*

---

## 1. System Overview

LIFEOS is **not** a chatbot with a dashboard or a simple expense tracker. It is a stateful intelligence platform that continuously:
1. Maintains a structured, deterministic representation of a user's financial and household state.
2. Learns from ingested transactions, bills, policies, warranties, and user inputs.
3. Detects meaningful changes and baseline deviations (anomalies).
4. Forecasts upcoming obligations (30/60/90-day horizon).
5. Simulates financial decisions (purchases, cash vs. EMI, income shocks) using deterministic calculations.
6. Powers an agentic tool-calling layer where answers are grounded in retrieved, verifiable evidence.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                       LIFEOS CLIENT                                     │
│                     (Next.js, TypeScript, Tailwind CSS, Recharts, Lucide)               │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │ HTTP / JSON REST APIs
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│                                    FASTAPI BACKEND ENGINE                               │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                 API Routing & Auth Layer                          │  │
│  │   /api/auth  •  /api/transactions  •  /api/dashboard  •  /api/events  •  /api/docs │  │
│  │   /api/assets • /api/loans • /api/subscriptions • /api/simulations • /api/agent   │  │
│  └────────────────────────────────────────┬──────────────────────────────────────────┘  │
│                                           │                                             │
│  ┌────────────────────────────────────────▼──────────────────────────────────────────┐  │
│  │                                  Intelligence Engines                             │  │
│  │  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌──────────────────────┐ │  │
│  │  │    Life State Engine    │ │   Event Engine (Horizon)│ │  Decision Simulator  │ │  │
│  │  │ (Deterministic Metrics) │ │  (30 / 60 / 90 Days)    │ │ (Buy / EMI / Stress) │ │  │
│  │  └─────────────────────────┘ └─────────────────────────┘ └──────────────────────┘ │  │
│  │  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌──────────────────────┐ │  │
│  │  │   Transaction Classifier│ │    Anomaly Detector     │ │  Expense Forecaster  │ │  │
│  │  │ (TF-IDF + Lin/Ens ML)   │ │  (Rolling Stats + IF)   │ │ (Roll/Trend/Seasonal)│ │  │
│  │  └─────────────────────────┘ └─────────────────────────┘ └──────────────────────┘ │  │
│  │  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌──────────────────────┐ │  │
│  │  │   Recurring Detector    │ │  Document Intelligence  │ │    Risk Engine       │ │  │
│  │  │ (Intervals + Merchant)  │ │ (Extraction + Provenance│ │ (Coverage / Runway)  │ │  │
│  │  └─────────────────────────┘ └─────────────────────────┘ └──────────────────────┘ │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Agentic Orchestrator & Tool Calling Layer                 │ │  │
│  │  │       (Typed Tools • Grounded Evidence • Read-Only Financial Guardrails)     │ │  │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────┬──────────────────────────────────────────┘  │
└───────────────────────────────────────────┼─────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│                                   PERSISTENCE & STORAGE                                 │
│  ┌───────────────────────────────────────────────────┐ ┌─────────────────────────────┐  │
│  │                 Relational Database               │ │         ChromaDB              │  │
│  │        MySQL 8.0 / SQLite (via SQLAlchemy)        │ │  (Semantic Document Search   │  │
│  │    Users, Accounts, Txns, Events, Alerts, State   │ │   & Retrieval-Augmented Gen)  │  │
│  └───────────────────────────────────────────────────┘ └─────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                   Local File Storage (Encrypted / Non-Public)                     │  │
│  │                    Uploaded PDFs, Statements, Warranties, Receipts                │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Principles

1. **Deterministic Core, Not LLM Hallucinations**:
   - Financial balances, surplus, runways, interest payments, loan EMIs, and cash projections are computed exclusively by deterministic Python algorithms using exact `Decimal` arithmetic.
   - The LLM never invents numbers, dates, or transactions. It interprets user intent, calls strictly typed tools, and explains results with citations.

2. **Per-User Isolation**:
   - Every single database read, write, query, and vector search is strictly filtered by the authenticated `user_id`. Multi-tenant security is enforced at the ORM query level and tested in automated suites.

3. **Database Dialect Independence**:
   - SQLAlchemy 2.0 declarative models use cross-compatible types (`Uuid`, `JSON`, `DECIMAL(14, 2)`).
   - Supports **MySQL 8.0** (`asyncmy` / `pymysql`), **PostgreSQL** (`asyncpg`), and zero-config local / test **SQLite** (`aiosqlite`).

4. **Auditability & Provenance**:
   - Every document extraction tracks source filename, confidence, and page reference.
   - Every state transition or alert records evidence and detection method.
   - An `AuditLog` captures state changes.

---

## 3. Database Schema Entities

The schema consists of 17 core relational entities:

1. **`users`**: Secure auth, hashed passwords (bcrypt), currency preferences (default INR `₹`), profile.
2. **`accounts`**: Bank accounts, credit cards, wallets, cash reserves with masked account numbers.
3. **`categories`**: System-seeded & user custom transaction categories (income vs expense, icons, colors).
4. **`transactions`**: Ingested entries with exact `Decimal` amounts, dates, debits/credits, classification confidence, anomaly flags, and raw metadata.
5. **`assets`**: Physical & digital property (vehicles, appliances, electronics) with purchase date, warranty expiry, current valuation, and TCO tracking.
6. **`loans`**: Personal, home, auto, education loans with principal, interest rate, EMI, tenure, and lender info.
7. **`insurance_policies`**: Health, term, motor, home policies with premium, frequency, renewal due dates, and policy numbers.
8. **`subscriptions`**: Recurring SaaS and media services with billing intervals and cancellation workflows.
9. **`documents`**: Uploaded statements, policies, agreements, warranties with extracted structured fields, verification status, and storage paths.
10. **`events`**: Future obligation horizon (30/60/90 days), status (`upcoming`, `due`, `completed`, `dismissed`, `overdue`), confidence, and source links.
11. **`goals`**: Savings targets, emergency funds, debt paydown with target dates and allocated balances.
12. **`alerts`**: System anomalies, upcoming deadlines, risk triggers with severity (`info`, `warning`, `critical`).
13. **`life_states`**: Historical immutable state snapshots (monthly income, expense, surplus, runway, debt ratio, risk score, changes from previous).
14. **`predictions`**: Category and aggregate forecasts with confidence intervals and model identifiers.
15. **`simulations`**: Stored purchase and stress-test scenarios with input parameters and outcome projections.
16. **`recurring_patterns`**: Detected recurring merchant patterns, cadence, and user confirmation state.
17. **`audit_logs`**: System audit trail of significant events, state changes, and user actions.

---

## 4. Key Engines

### A. Life State Engine (`app.services.life_state`)
- Deterministically computes the user's instantaneous financial posture:
  - `monthly_income`, `monthly_expenses`, `net_surplus`
  - `liquid_cash`, `total_debt`, `debt_to_income_ratio`
  - `emergency_runway_months` = Liquid Cash / Average Monthly Essential Expenses
  - `financial_risk_score` (0–100 explainable scale)
- Computes state differentials (`What Changed?`) comparing snapshot $T$ with snapshot $T-1$.

### B. Event Horizon Engine (`app.services.event_engine`)
- Consolidates upcoming EMIs, insurance renewals, recurring subscriptions, warranty expirations, and user deadlines.
- Aggregates rolling cash obligations for **30-day**, **60-day**, and **90-day** outlooks.
- Powers the signature feature: **"What am I forgetting?"**.

### C. ML & Transaction Intelligence Engine (`app.ml`)
- **Classifier**: Hybrid rule-based + TF-IDF Logistic Regression pipeline categorizing transaction narrations with confidence scoring.
- **Recurring Detector**: Identifies merchant regularity, frequency (weekly, monthly, annual), amount variance, and predicts next billing dates.
- **Anomaly Detector**: Rolling $z$-score and statistical quantile fences + Isolation Forest for outlier spending.

### D. Decision & Stress Simulator (`app.services.simulator`)
- **Purchase Affordability**: Evaluates buying an asset via outright cash vs. EMI across delayed timelines (Now, +3 months, +6 months). Calculates cash balance trajectory against minimum safety buffers.
- **Stress Testing**: Simulates income cuts (-20%, -50%, 0%), emergency expenses (medical/vehicle), or sudden rate hikes, reporting runway shrinkage.

### E. Agentic Orchestration Layer (`app.agent`)
- Implements 13 typed tools (`get_life_state`, `get_transactions`, `simulate_purchase`, `forecast_expenses`, `detect_anomalies`, `search_documents`, etc.).
- Enforces strict read-only financial safety guardrails and evidence-backed citations.
