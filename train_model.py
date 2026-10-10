import argparse
import json
from datetime import datetime, timezone
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
LEGACY_ARCHIVE_DIR = DATA_DIR / "legacy_uci_archive"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# app.py always loads models/heart_model.joblib (the default, Cleveland-only
# model). Non-default dataset variants are saved under their own filenames so
# they never silently replace the model the app is actually serving.
DEFAULT_DATASET = "cleveland"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]

DATASET_LABELS = {
    "cleveland": "UCI Heart Disease Repository (Cleveland database)",
    "combined": (
        "UCI Heart Disease Repository "
        "(Cleveland, Hungarian, Switzerland, Long Beach VA - combined)"
    ),
}

# The four site files that make up the UCI Heart Disease archive. Each one
# uses the same 14-column schema (13 features + num) with "?" for missing
# values. Only Cleveland is used by default; "combined" trains on all four
# for a larger (~920 row), noisier, multi-site dataset.
COMBINED_SOURCE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def load_cleveland_dataset():
    """Fetch the 303-row Cleveland dataset from the UCI repository."""
    print("Downloading the UCI Heart Disease dataset (Cleveland)...")
    dataset = fetch_ucirepo(id=45)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()
    X.columns = [str(c).strip().lower() for c in X.columns]

    # Save raw dataset to data/raw/ for reproducibility
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / "heart.csv"
    combined_df = pd.concat([pd.DataFrame(dataset.data.features), y], axis=1)
    combined_df.to_csv(raw_path, index=False)
    print(f"Raw dataset saved to: {raw_path}")

    return X, y


def load_combined_dataset():
    """Load and concatenate all four UCI heart-disease site files.

    These files are archived in data/legacy_uci_archive/ (same schema as the
    Cleveland file, "?" for missing values). Switzerland's cholesterol column
    is entirely 0 because the test wasn't performed at that site, so a
    literal 0 is treated as missing rather than a real reading, for every
    source (a true serum cholesterol of 0 mg/dL isn't physiologically
    possible either way).
    """
    columns = FEATURES + ["num"]
    frames = []

    for source, filename in COMBINED_SOURCE_FILES.items():
        path = LEGACY_ARCHIVE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected archived dataset file not found: {path}"
            )
        frame = pd.read_csv(path, header=None, names=columns, na_values="?")
        frame["source"] = source
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data.loc[data["chol"] == 0, "chol"] = np.nan

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["num"], errors="coerce")

    print(f"Combined {len(frames)} sites: "
          + ", ".join(f"{s}={len(f)}" for s, f in zip(COMBINED_SOURCE_FILES, frames)))

    # Save raw combined dataset to data/raw/ for reproducibility
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / "heart_combined.csv"
    data.to_csv(raw_path, index=False)
    print(f"Raw combined dataset saved to: {raw_path}")

    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train the heart disease prediction model."
    )
    parser.add_argument(
        "--dataset",
        choices=["cleveland", "combined"],
        default=DEFAULT_DATASET,
        help=(
            "'cleveland' (default): 303-row Cleveland subset via ucimlrepo, "
            "saved to models/heart_model.joblib (the model app.py serves). "
            "'combined': all four UCI sites (~920 rows, noisier, more missing "
            "data) from data/legacy_uci_archive/, saved to "
            "models/heart_model_combined.joblib so it never replaces the "
            "default model app.py loads."
        ),
    )
    args = parser.parse_args()

    # Non-default variants get their own filenames so a run never silently
    # overwrites the model/reports that app.py and predict_sample.py expect.
    suffix = "" if args.dataset == DEFAULT_DATASET else f"_{args.dataset}"
    model_path = MODEL_DIR / f"heart_model{suffix}.joblib"
    eval_json_path = REPORTS_DIR / f"model_evaluation{suffix}.json"
    eval_txt_path = REPORTS_DIR / f"model_evaluation{suffix}.txt"

    if args.dataset == "combined":
        X, y = load_combined_dataset()
    else:
        X, y = load_cleveland_dataset()

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
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Target 0", "Target 1"],
        zero_division=0,
    )
    print(report_text)

    report_dict = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Target 0", "Target 1"],
        zero_division=0,
        output_dict=True,
    )

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": args.dataset,
        "dataset_label": DATASET_LABELS[args.dataset],
        "instance_count": len(X),
        "features": FEATURES,
        "target_definition": "Original UCI target 0 versus target values 1-4",
        "test_size": 0.20,
        "random_state": 42,
        "test_set_size": len(y_test),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc if y_test.nunique() == 2 else None,
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": report_dict,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write("AI Heart Disease Prediction System - Model Evaluation\n")
        f.write(f"Dataset: {evaluation['dataset_label']}\n")
        f.write(f"Generated at: {evaluation['generated_at']}\n")
        f.write(f"Test set size: {evaluation['test_set_size']}\n\n")
        f.write(f"Accuracy:  {accuracy:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall:    {recall:.3f}\n")
        f.write(f"F1-score:  {f1:.3f}\n")
        if evaluation["metrics"]["roc_auc"] is not None:
            f.write(f"ROC-AUC:   {evaluation['metrics']['roc_auc']:.3f}\n")
        f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        f.write(f"{np.array(evaluation['confusion_matrix'])}\n")
        f.write("\nClassification report:\n")
        f.write(report_text)

    print(f"\nEvaluation reports saved to: {eval_json_path} and {eval_txt_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the complete pipeline, including preprocessing.
    joblib.dump(
        {
            "pipeline": model,
            "features": FEATURES,
            "dataset_source": args.dataset,
            "dataset_label": DATASET_LABELS[args.dataset],
            "instance_count": len(X),
            "target_definition": (
                "Original UCI target 0 versus target values 1-4"
            ),
            "test_size": 0.20,
            "random_state": 42,
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")
    if args.dataset != DEFAULT_DATASET:
        print(
            f"Note: this is a '{args.dataset}' model, saved separately from "
            f"the default models/heart_model.joblib that app.py loads."
        )


if __name__ == "__main__":
    main()