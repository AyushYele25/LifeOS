"""
LIFEOS — Dataset Explorer
Quick profile of downloaded Kaggle datasets to understand
what we have for ML pipeline development.
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("  LIFEOS DATASET EXPLORER")
print("=" * 70)

# ─── Dataset 1: Personal Finance Data ───────────────────────────────
print("\n\n📊 DATASET 1: Personal Finance Dataset")
print("─" * 50)

pf = pd.read_csv(os.path.join(DATA_DIR, "kaggle", "personal_finance", "Personal_Finance_Dataset.csv"))
print(f"Shape: {pf.shape[0]} rows × {pf.shape[1]} columns")
print(f"\nColumns: {list(pf.columns)}")
print(f"\nDtypes:\n{pf.dtypes}")
print(f"\nSample rows:")
print(pf.head(10).to_string())
print(f"\n📂 Categories ({pf['Category'].nunique()} unique):")
print(pf['Category'].value_counts().to_string())
print(f"\n💰 Type distribution:")
print(pf['Type'].value_counts().to_string())
print(f"\n📅 Date range: {pf['Date'].min()} → {pf['Date'].max()}")
print(f"💵 Amount stats:\n{pf['Amount'].describe().to_string()}")

# ─── Dataset 2: UPI Transactions ────────────────────────────────────
print("\n\n📊 DATASET 2: UPI Transactions")
print("─" * 50)

tx = pd.read_csv(os.path.join(DATA_DIR, "kaggle", "transactions", "transactions.csv"))
print(f"Shape: {tx.shape[0]} rows × {tx.shape[1]} columns")
print(f"\nColumns: {list(tx.columns)}")
print(f"\nDtypes:\n{tx.dtypes}")
print(f"\nSample rows:")
print(tx.head(10).to_string())
print(f"\n📊 Status distribution:")
print(tx['Status'].value_counts().to_string())
print(f"\n📅 Date range: {tx['Timestamp'].min()} → {tx['Timestamp'].max()}")
print(f"💵 Amount stats:\n{tx['Amount (INR)'].describe().to_string()}")
print(f"\n👤 Unique senders: {tx['Sender Name'].nunique()}")
print(f"👤 Unique receivers: {tx['Receiver Name'].nunique()}")

# ─── Usefulness Assessment ──────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  USEFULNESS FOR LIFEOS ML PIPELINE")
print("=" * 70)

print("""
┌─────────────────────────────────┬─────────────────────────────────────────┐
│ ML Component                    │ Dataset Coverage                        │
├─────────────────────────────────┼─────────────────────────────────────────┤
│ Transaction Classifier          │ ✅ Personal Finance (labeled categories)│
│ Anomaly Detector                │ ⚠️  Need fraud-labeled data (missing)   │
│ Recurring Detector              │ ✅ Personal Finance (multi-year data)   │
│ Expense Forecaster              │ ✅ Personal Finance (4yr time series)   │
│ UPI Pattern Analysis            │ ✅ UPI Transactions (Indian names/VPAs) │
│ Document Intelligence           │ ❌ Not covered (need PDF datasets)      │
│ Risk Scoring                    │ ⚠️  Partial (need benchmarks)           │
└─────────────────────────────────┴─────────────────────────────────────────┘
""")
