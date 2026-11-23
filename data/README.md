# LIFEOS — Datasets & Synthetic Data Documentation

This directory houses synthetic test datasets, evaluation ground truth, and reference datasets used by the LIFEOS intelligence pipelines.

---

## 1. Synthetic Data Generator (`synthetic_data.py`)

The primary data source for LIFEOS is a deterministic, reproducible synthetic data generator designed to simulate realistic Indian household finances without relying on third-party API credentials or compromising personal data.

### Generation Commands

```bash
# 1. Generate full master dataset (100 users, 12 months, seed 42)
python backend/app/services/synthetic_data.py --users 100 --months 12 --seed 42 --output-dir data/synthetic

# 2. Quick development dataset (5 users, 3 months)
python backend/app/services/synthetic_data.py --users 5 --months 3 --seed 42 --output-dir data/synthetic

# 3. Generate and directly seed the development database
python backend/app/services/synthetic_data.py --users 10 --months 6 --seed 42 --seed-db
```

### Generated Artifacts in `data/synthetic/`

| File | Format | Records | Description |
| :--- | :--- | :--- | :--- |
| `synthetic_dataset.json` | JSON | Complete Graph | Nested JSON representation of users, accounts, assets, loans, insurance, subscriptions, events, and transactions. |
| `synthetic_transactions.csv` | CSV | 29,731 rows | Flat CSV with `id`, `user_id`, `account_id`, `date`, `description`, `amount`, `transaction_type`, `merchant`, `category`, `is_recurring`, `is_anomaly`. |
| `anomalies_ground_truth.csv` | CSV | 75 rows | Ground-truth labels for injected anomalies including baseline expectation, observed anomaly, and evidence string. |

---

## 2. Indian Household Persona Modeling

The generator simulates realistic Indian urban households with distinct demographic parameters:

- **Income Profiles**: Salaries ranging from ₹35,000/mo to ₹220,000/mo (junior, mid, senior, managerial tiers).
- **Banking Context**: Multi-account setups with Indian institutions (HDFC, SBI, ICICI, Axis, Kotak), primary savings accounts, masked credit cards, and UPI identifiers.
- **Fixed Monthly Outflows**:
  - Rent: 20%–32% of monthly income.
  - Vehicle / Personal Loan EMIs: Deducted via NACH on the 10th of every month.
  - Subscriptions: Netflix, Spotify, Amazon Prime, Hotstar, Cult.fit, ACT / Airtel Broadband.
  - Utilities: Electricity (BESCOM, Tata Power, Adani), Gas (IGL, MGL), Mobile (Jio, Airtel).
- **Variable Living Expenses**:
  - Groceries (Blinkit, Zepto, BigBasket, Instamart, DMart).
  - Food & Dining (Swiggy, Zomato, local restaurants).
  - Transit & Fuel (Uber, Ola, Shell, Indian Oil, Namma Metro).
- **Seasonal Spending Variations**:
  - **Summer Utility Spike** (April–June): Electricity consumption spikes by +40% to +80% due to air conditioning.
  - **Festive Diwali Spike** (October–November): Discretionary retail shopping increases significantly.

---

## 3. Injected Anomalies & Ground-Truth Labels

To rigorously evaluate the **Anomaly Detection Engine** (Phase 5), the generator injects controlled deviations with known ground truth:

1. **Utility Spikes (`utility_spike`)**:
   - An electricity bill suddenly surges 3.8× over the user's historical rolling baseline (e.g. ₹6,800 vs ₹1,800).
2. **Sudden Salary Drops (`salary_drop`)**:
   - An unexpected 45% reduction in monthly corporate salary credit.
3. **Duplicate Transactions (`duplicate_transaction`)**:
   - An identical merchant debit for the exact same amount executed within minutes.
4. **Unexpected Large Purchases (`unexpected_large_purchase`)**:
   - An out-of-band high-ticket retail transaction (e.g. ₹92,000 at Apple Store) exceeding typical shopping averages by >700%.

Ground-truth files report:
```csv
user_id,date,type,expected,observed,evidence
0eae415a...,2026-08-14,utility_spike,1650.0,6270.0,"Electricity bill ₹6270.00 is 280% higher than historical average of ₹1650.00"
```

---

## 4. Synthetic Documents Ground Truth

The generator generates synthetic document records simulating physical/digital paperwork:
- **Vehicle Loan Agreements**: Principal, lender, interest rate (8.75%), EMI, 48-month tenure, due day.
- **Health Insurance Schedules**: Provider (HDFC ERGO, Star Health), policy number, ₹10,00,000 sum insured, annual premium, expiry/renewal date.
- **Warranties & Bills**: Products, purchase dates, warranty duration, and expiration horizons.

---

## 5. Kaggle External Datasets

Located in `data/kaggle/`:

### A. Personal Finance Dataset
- **Path**: `data/kaggle/personal_finance/Personal_Finance_Dataset.csv`
- **Volume**: 1,500 rows × 5 columns (`Date`, `Transaction Description`, `Category`, `Amount`, `Type`).
- **Time Range**: 2020-01-02 to 2024-12-29 (4 years).
- **Categories**: Rent, Travel, Utilities, Health & Fitness, Shopping, Food & Drink, Salary, Entertainment, Investment, Other.
- **Use in LIFEOS**: Training baseline for TF-IDF + Logistic Regression transaction classifier and multi-year time-series forecasting.

### B. Indian UPI Transactions Dataset
- **Path**: `data/kaggle/transactions/transactions.csv`
- **Volume**: 1,000 rows × 8 columns (`Transaction ID`, `Timestamp`, `Sender Name`, `Sender UPI ID`, `Receiver Name`, `Receiver UPI ID`, `Amount (INR)`, `Status`).
- **VPAs**: `@okaxis`, `@okybl`, `@okicici`, `@okhdfcbank`, `@oksbi`.
- **Use in LIFEOS**: UPI parsing, Indian VPA intelligence, counterparty extraction, and failure-rate monitoring.
