"""
LIFEOS Transaction Classification Pipeline.

Hybrid rule-based + TF-IDF Logistic Regression classifier with calibrated confidence scoring.
Categorizes transaction descriptions into standard LIFEOS categories.
"""

import os
import re
import joblib
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from app.config import settings

# High-precision keyword dictionary for deterministic classification
RULE_PATTERNS = {
    "Salary": [r"\bsalary\b", r"\bpayroll\b", r"\bach credit\b", r"\bwages\b", r"\bstipend\b"],
    "Insurance": [r"\binsurance\b", r"\blic\b", r"\blic of india\b", r"\bstar health\b", r"\bhdfc ergo\b", r"\bicici lombard\b", r"\bpolicybazaar\b", r"\bmax bupa\b", r"\bniva bupa\b", r"\btata aig\b", r"\bcare health\b"],
    "Rent": [r"\brent\b", r"\blandlord\b", r"\bhouse rent\b", r"\bflat rent\b"],
    "EMI & Loans": [r"\bemi\b", r"\bloan\b", r"\bhome loan\b", r"\bauto loan\b", r"\bpersonal loan\b", r"\bnach emi\b", r"\bnach loan\b", r"\bcar loan\b"],
    "Electricity": [r"\belectricity\b", r"\bbescom\b", r"\btata power\b", r"\badani elec\b", r"\bbses\b", r"\bpower corp\b"],
    "Utilities": [r"\bgas bill\b", r"\bindraprastha gas\b", r"\bmahanagar gas\b", r"\bwater bill\b", r"\bbill desk\b"],
    "Mobile & Internet": [r"\bairtel\b", r"\bjio\b", r"\bvodafone\b", r"\bvi\b", r"\bbroadband\b", r"\bfiber\b", r"\bact fiber\b"],
    "Subscriptions": [r"\bnetflix\b", r"\bspotify\b", r"\bprime\b", r"\bhotstar\b", r"\byoutube\b", r"\bcult\.fit\b", r"\bapple\.com/bill\b"],
    "Food & Dining": [r"\bswiggy\b", r"\bzomato\b", r"\bmcdonald\b", r"\bstarbucks\b", r"\bdominos\b", r"\bhaldiram\b", r"\bcoffee\b", r"\brestaurant\b", r"\bcafe\b"],
    "Groceries": [r"\bblinkit\b", r"\bzepto\b", r"\binstamart\b", r"\bbigbasket\b", r"\bdmart\b", r"\bkirana\b", r"\bsupermarket\b", r"\bgrocery\b"],
    "Fuel": [r"\bfuel\b", r"\bpetrol\b", r"\bdiesel\b", r"\bindian oil\b", r"\bbharat petroleum\b", r"\bbharat petro\b", r"\bhpcl\b", r"\bshell\b"],
    "Transportation": [r"\buber\b", r"\bola\b", r"\bblusmart\b", r"\bmetro\b", r"\brailway\b", r"\birctc\b", r"\bauto rickshaw\b", r"\bautorickshaw\b", r"\bauto-rickshaw\b", r"\bcab\b", r"\btaxi\b"],
    "Shopping": [r"\bamazon\b", r"\bflipkart\b", r"\bmyntra\b", r"\bzara\b", r"\buniqlo\b", r"\bcroma\b", r"\bajio\b", r"\bretail\b", r"\bclothing\b"],
    "Health & Medical": [r"\bpharmacy\b", r"\bapollo\b", r"\b1mg\b", r"\bmedplus\b", r"\bhospital\b", r"\bclinic\b", r"\bdr\.\b", r"\blab\b"],
    "Entertainment": [r"\bbookmyshow\b", r"\bpvr\b", r"\binox\b", r"\bcinema\b", r"\bmovie\b", r"\bsteam\b", r"\bgaming\b"],
    "Maintenance": [r"\burban company\b", r"\bmaintenance\b", r"\belectrician\b", r"\bplumber\b", r"\bsociety\b"],
    "Investments": [r"\bzerodha\b", r"\bgroww\b", r"\bmf\b", r"\bmutual fund\b", r"\bshares\b", r"\bsecurities\b", r"\bkuvera\b"],
}


def preprocess_text(text: str) -> str:
    """Clean and normalize transaction description."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.lower()
    # Remove technical transaction prefixes/suffixes
    cleaned = re.sub(r"^(upi/|pos/|ach credit —|nach debit —|e-com/|bbps |pos/e-com recurring —)", "", cleaned)
    cleaned = re.sub(r"ref-\d+", "", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class TransactionClassifier:
    """Hybrid Classifier combining rule-based heuristics and TF-IDF Logistic Regression."""

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.categories: List[str] = list(RULE_PATTERNS.keys()) + ["Miscellaneous"]
        self.confidence_threshold: float = settings.CLASSIFICATION_CONFIDENCE_THRESHOLD

    def match_rules(self, text: str) -> Optional[Tuple[str, float]]:
        """Check deterministic keyword rules."""
        cleaned = text.lower()
        for category, patterns in RULE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, cleaned):
                    return category, 0.96
        return None

    def fit(self, descriptions: List[str], labels: List[str]):
        """Train the TF-IDF + Logistic Regression pipeline."""
        processed = [preprocess_text(d) for d in descriptions]
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=10000,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=1.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        self.model.fit(processed, labels)
        self.categories = list(self.model.classes_)
        return self

    def predict(self, description: str) -> Tuple[str, float, bool]:
        """
        Classify a transaction description.
        Returns:
            (category_name, confidence_score, is_confident)
        """
        # 1. Rule Match check first
        rule_match = self.match_rules(description)
        if rule_match:
            cat, conf = rule_match
            return cat, conf, conf >= self.confidence_threshold

        # 2. ML Prediction if model is loaded
        if self.model is not None:
            cleaned = preprocess_text(description)
            if cleaned:
                probs = self.model.predict_proba([cleaned])[0]
                best_idx = np.argmax(probs)
                category = self.model.classes_[best_idx]
                confidence = float(probs[best_idx])
                return category, confidence, confidence >= self.confidence_threshold

        # 3. Default fallback
        return "Miscellaneous", 0.35, False

    def save(self, filepath: str):
        """Serialize model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "categories": self.categories}, filepath)

    def load(self, filepath: str) -> bool:
        """Load serialized model from disk."""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data.get("model")
            self.categories = data.get("categories", self.categories)
            return True
        return False


_global_classifier: Optional[TransactionClassifier] = None


def get_classifier() -> TransactionClassifier:
    """Singleton getter for TransactionClassifier."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = TransactionClassifier()
        model_path = os.path.join(settings.ML_MODEL_DIR, "transaction_classifier.joblib")
        if not os.path.isabs(model_path):
            backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            model_path = os.path.join(backend_dir, settings.ML_MODEL_DIR, "transaction_classifier.joblib")
        _global_classifier.load(model_path)
    return _global_classifier
