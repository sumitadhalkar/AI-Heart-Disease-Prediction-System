from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]


def main():
    print("Downloading the UCI Heart Disease dataset...")
    dataset = fetch_ucirepo(id=45)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Normalize feature names and ensure a consistent feature schema.
    X.columns = [str(c).strip().lower() for c in X.columns]

    missing_features = set(FEATURES) - set(X.columns)
    if missing_features:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_features}. "
            f"Available columns: {list(X.columns)}"
        )

    X = X[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # UCI target: 0 = no disease; 1–4 = presence of disease
    # in the original dataset's target coding.
    valid = y.notna()
    X = X.loc[valid].copy()
    y = (y.loc[valid] > 0).astype(int)

    if y.nunique() != 2:
        raise ValueError("Expected two target classes after conversion.")

    print(f"Usable records: {len(X)}")
    print("Target counts:")
    print(y.value_counts().sort_index().rename(
        index={0: "Target 0", 1: "Target 1"}
    ))

    # Save raw dataset to data/raw/ for reproducibility
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Convert dataset to DataFrame and save
    X_df = pd.DataFrame(dataset.data.features)
    y_df = dataset.data.targets.iloc[:, 0]
    raw_path = RAW_DATA_DIR / "heart.csv"
    # Save combined data
    combined = pd.concat([X_df, y_df], axis=1)
    combined.to_csv(raw_path, index=False)
    print(f"Raw dataset saved to: {raw_path}")

    # Split before fitting preprocessing to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    print("\n--- Held-out test evaluation ---")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, scores)
        print(f"ROC-AUC:   {auc:.3f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Target 0", "Target 1"],
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the complete pipeline, including preprocessing.
    joblib.dump(
        {
            "pipeline": model,
            "features": FEATURES,
            "target_definition": (
                "Original UCI target 0 versus target values 1-4"
            ),
            "test_size": 0.20,
            "random_state": 42,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()