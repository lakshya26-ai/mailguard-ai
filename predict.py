"""
predict.py
----------
Loads the trained TF-IDF vectorizer + Naive Bayes model and exposes a single
`classify_email()` helper that the Streamlit app calls.

This module deliberately has no Streamlit imports -- it is pure Python so it
can also be unit-tested or reused from a CLI / API without pulling in the
web layer.
"""

import json
from pathlib import Path
from typing import TypedDict

import joblib

from src.train_model import clean_text

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


class ClassificationResult(TypedDict):
    label: str          # "spam" or "ham"
    is_spam: bool
    confidence: float   # 0-100, confidence in the predicted label
    spam_probability: float   # 0-100
    ham_probability: float    # 0-100


class ModelNotTrainedError(RuntimeError):
    """Raised when the app is launched before the model has been trained."""


_model = None
_vectorizer = None
_metrics_cache = None


def _load_artifacts():
    """Lazily load and cache the model + vectorizer (loaded once per process)."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise ModelNotTrainedError(
                "Model artifacts not found. Run `python src/train_model.py` "
                "from the project root before starting the app."
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def get_metrics() -> dict:
    """Return the saved evaluation metrics dict (accuracy, precision, etc.)."""
    global _metrics_cache
    if _metrics_cache is None:
        if not METRICS_PATH.exists():
            raise ModelNotTrainedError(
                "Metrics file not found. Run `python src/train_model.py` first."
            )
        with open(METRICS_PATH) as f:
            _metrics_cache = json.load(f)
    return _metrics_cache


def classify_email(subject: str, body: str) -> ClassificationResult:
    """Classify a subject + body pair as spam or ham (not spam).

    Raises:
        ValueError: if there is no meaningful text to classify.
        ModelNotTrainedError: if the model has not been trained yet.
    """
    model, vectorizer = _load_artifacts()

    combined_text = f"{subject or ''} {body or ''}".strip()
    if not combined_text:
        raise ValueError("No text provided to classify.")

    cleaned = clean_text(combined_text)
    if not cleaned:
        raise ValueError(
            "The provided text contains no usable words after cleaning."
        )

    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    classes = list(model.classes_)  # e.g. ['ham', 'spam']

    ham_idx = classes.index("ham")
    spam_idx = classes.index("spam")

    ham_prob = float(proba[ham_idx]) * 100
    spam_prob = float(proba[spam_idx]) * 100

    label = "spam" if spam_prob >= ham_prob else "ham"
    confidence = spam_prob if label == "spam" else ham_prob

    return ClassificationResult(
        label=label,
        is_spam=(label == "spam"),
        confidence=round(confidence, 1),
        spam_probability=round(spam_prob, 1),
        ham_probability=round(ham_prob, 1),
    )
