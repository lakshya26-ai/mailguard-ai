"""
train_model.py
---------------
Trains the MailGuard AI spam classifier.

Pipeline:
    raw text -> clean_text() -> TF-IDF vectorizer -> Multinomial Naive Bayes

Running this script will:
    1. Load data/spam.csv
    2. Clean + split the data (train/test)
    3. Fit a TF-IDF vectorizer on the training text
    4. Train a MultinomialNB classifier
    5. Evaluate on the held-out test set (accuracy, precision, recall, F1,
       confusion matrix)
    6. Persist the vectorizer, model, and a small metrics.json file to /model
       so the Streamlit app never has to retrain on startup.

Usage:
    python src/train_model.py
"""

import json
import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "spam.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42

# A short, deliberately common list of English stopwords. Kept local (no NLTK
# download step required) so the project has zero network dependency at
# train/run time beyond pip installs.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}


def clean_text(text: str) -> str:
    """Normalize raw email/SMS text before vectorization.

    Lower-cases, strips URLs/emails/numbers down to placeholder tokens,
    removes punctuation, and drops stopwords. Keeping this function here
    (rather than only inline in a notebook) means predict.py can import the
    *exact* same transformation used at train time.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


REAL_PATH = ROOT_DIR / "data" / "real_world_examples.csv"
SITE_MODEL_PATH = ROOT_DIR / "docs" / "model.js"
REAL_WORLD_WEIGHT = 4  # oversampling factor for the hand-written email examples


def load_dataset() -> pd.DataFrame:
    """SMS corpus + real-world email examples + any data/extra_*.csv files
    (label,message columns) you drop in, e.g. SpamAssassin or Enron exports."""
    frames = [pd.read_csv(DATA_PATH)[["label", "message"]].assign(source="sms")]
    if REAL_PATH.exists():
        frames.append(pd.read_csv(REAL_PATH).assign(source="email"))
    for extra in sorted((ROOT_DIR / "data").glob("extra_*.csv")):
        frames.append(pd.read_csv(extra)[["label", "message"]].assign(source="extra"))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["message", "label"]).drop_duplicates(subset=["message"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    df["clean_message"] = df["message"].astype(str).apply(clean_text)
    df = df[df["clean_message"].str.len() > 0]
    return df.reset_index(drop=True)


def export_for_web(model, vectorizer, metrics: dict) -> None:
    """Dump the fitted model as plain numbers so the static site can run inference in JS."""
    payload = {
        "classes": list(model.classes_),
        "vocab": vectorizer.get_feature_names_out().tolist(),
        "idf": [round(float(x), 5) for x in vectorizer.idf_],
        "prior": [round(float(x), 5) for x in model.class_log_prior_],
        "lp": [[round(float(x), 5) for x in row] for row in model.feature_log_prob_],
        "stop": sorted(STOPWORDS),
        "metrics": metrics,
    }
    SITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_MODEL_PATH.write_text(
        "window.MG_MODEL=" + json.dumps(payload, separators=(",", ":")) + ";"
    )


def train_and_evaluate() -> dict:
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    # Real email examples are scarce next to ~5k SMS rows, so upweight them (train split only).
    boost = train_df[train_df["source"] != "sms"]
    train_df = pd.concat([train_df] + [boost] * (REAL_WORLD_WEIGHT - 1), ignore_index=True)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(train_df["clean_message"])
    X_test_vec = vectorizer.transform(test_df["clean_message"])
    y_train, y_test = train_df["label"], test_df["label"]

    model = MultinomialNB(alpha=0.3)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="spam"), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="spam"), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label="spam"), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist(),
        "confusion_matrix_labels": ["ham", "spam"],
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "total_size": int(len(df)),
        "email_examples": int((df["source"] != "sms").sum()),
        "model_type": "Multinomial Naive Bayes",
        "vectorizer_type": "TF-IDF (unigrams + bigrams, 5000 features)",
        "dataset_name": "SMS Spam Collection + real-world email examples",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    export_for_web(model, vectorizer, metrics)
    return metrics


if __name__ == "__main__":
    m = train_and_evaluate()
    print("Training complete.\n")
    print(f"  Dataset size     : {m['total_size']} messages")
    print(f"  Train / Test     : {m['train_size']} / {m['test_size']}")
    print(f"  Accuracy         : {m['accuracy'] * 100:.2f}%")
    print(f"  Precision (spam) : {m['precision'] * 100:.2f}%")
    print(f"  Recall (spam)    : {m['recall'] * 100:.2f}%")
    print(f"  F1 score (spam)  : {m['f1_score'] * 100:.2f}%")
    print(f"  Confusion matrix : {m['confusion_matrix']}")
    print(f"\nSaved web model  -> {SITE_MODEL_PATH}")
    print(f"Saved model      -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved metrics    -> {METRICS_PATH}")
