"""
LIFEOS ML — Transaction Classifier Training Script.

Trains hybrid TF-IDF + Logistic Regression model on combined synthetic & Kaggle datasets.
Evaluates on a held-out test set and saves the serialized model and evaluation metrics.
"""

import json
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ml.classifier import TransactionClassifier
from app.config import settings

# Mapping from external datasets to standard LIFEOS categories
CATEGORY_NORMALIZATION = {
    "food & drink": "Food & Dining",
    "food & dining": "Food & Dining",
    "dining": "Food & Dining",
    "groceries": "Groceries",
    "grocery": "Groceries",
    "rent": "Rent",
    "housing": "Rent",
    "salary": "Salary",
    "income": "Salary",
    "utilities": "Utilities",
    "electricity": "Electricity",
    "shopping": "Shopping",
    "entertainment": "Entertainment",
    "travel": "Transportation",
    "transportation": "Transportation",
    "fuel": "Fuel",
    "health & fitness": "Health & Medical",
    "health & medical": "Health & Medical",
    "investment": "Investments",
    "investments": "Investments",
    "subscriptions": "Subscriptions",
    "subscription": "Subscriptions",
    "emi & loans": "EMI & Loans",
    "other": "Miscellaneous",
    "miscellaneous": "Miscellaneous",
}


def load_and_prepare_data(workspace_root: str):
    """Load and unify training data from synthetic and Kaggle datasets."""
    descriptions = []
    labels = []

    # 1. Synthetic dataset
    synthetic_csv = os.path.join(workspace_root, "data", "synthetic", "synthetic_transactions.csv")
    if os.path.exists(synthetic_csv):
        print(f"📖 Loading synthetic dataset: {synthetic_csv}")
        df_syn = pd.read_csv(synthetic_csv)
        for _, row in df_syn.iterrows():
            desc = str(row.get("description", "")).strip()
            cat = str(row.get("category", "")).strip().lower()
            norm_cat = CATEGORY_NORMALIZATION.get(cat, "Miscellaneous")
            if desc and norm_cat:
                descriptions.append(desc)
                labels.append(norm_cat)

    # 2. Kaggle Personal Finance dataset
    kaggle_pf = os.path.join(workspace_root, "data", "kaggle", "personal_finance", "Personal_Finance_Dataset.csv")
    if os.path.exists(kaggle_pf):
        print(f"📖 Loading Kaggle Personal Finance dataset: {kaggle_pf}")
        df_kg = pd.read_csv(kaggle_pf)
        for _, row in df_kg.iterrows():
            desc = str(row.get("Transaction Description", "")).strip()
            cat = str(row.get("Category", "")).strip().lower()
            norm_cat = CATEGORY_NORMALIZATION.get(cat, "Miscellaneous")
            if desc and norm_cat:
                descriptions.append(desc)
                labels.append(norm_cat)

    print(f"📊 Total training records assembled: {len(descriptions):,}")
    return descriptions, labels


def train_and_evaluate():
    """Execute training pipeline, compute metrics, and save model."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workspace_root = os.path.abspath(os.path.join(backend_dir, ".."))

    descriptions, labels = load_and_prepare_data(workspace_root)
    if not descriptions:
        raise ValueError("No training data found! Run synthetic generator first.")

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        descriptions, labels, test_size=0.20, random_state=42, stratify=labels
    )

    print(f"⚙️ Training split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")

    classifier = TransactionClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate on held-out test split
    print("\n🔍 Evaluating model on held-out test split...")
    y_pred = []
    for text in X_test:
        pred_cat, _, _ = classifier.predict(text)
        y_pred.append(pred_cat)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print("=" * 65)
    print("  TRANSACTION CLASSIFIER EVALUATION REPORT")
    print("=" * 65)
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nPer-Category Breakdown:")
    for cat, metrics in report_dict.items():
        if isinstance(metrics, dict):
            print(f"  {cat:20} P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")

    # Save serialized model
    model_dir = os.path.join(backend_dir, "ml_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "transaction_classifier.joblib")
    classifier.save(model_path)
    print(f"\n💾 Model serialized to: {model_path}")

    # Save metrics JSON
    metrics_path = os.path.join(model_dir, "classifier_metrics.json")
    metrics_summary = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_samples": len(X_test),
        "train_samples": len(X_train),
        "per_category": report_dict,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"📈 Evaluation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train_and_evaluate()
