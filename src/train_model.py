"""
train_model.py
---------------
Loads the (feature-extracted) dataset, trains a RandomForestClassifier to
distinguish phishing vs. legitimate URLs, evaluates it, and saves:
  - model/phishing_model.pkl      (trained sklearn pipeline/model)
  - model/feature_columns.pkl     (exact feature order the model expects)
  - model/metrics.json            (accuracy/precision/recall/f1/confusion matrix)

Run from the project root:
    python src/train_model.py
"""

import os
import json
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
from features import FEATURE_NAMES  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "phishing_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "model")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["label"]
    return X, y, df


def train_and_evaluate(X, y, model_type="random_forest"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type": model_type,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Metrics:", json.dumps(metrics, indent=2))

    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        metrics["feature_importances"] = [
            {"feature": f, "importance": round(float(i), 4)} for f, i in importances
        ]
        print("\nTop features:")
        for f, i in importances[:8]:
            print(f"  {f:28s} {i:.4f}")

    return model, metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, _ = load_data()

    model, metrics = train_and_evaluate(X, y, model_type="random_forest")

    joblib.dump(model, os.path.join(MODEL_DIR, "phishing_model.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}/phishing_model.pkl")
    print(f"Saved metrics to {MODEL_DIR}/metrics.json")


if __name__ == "__main__":
    main()
