from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.joblib"

app = Flask(__name__)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]

# Inclusive input limits for basic form validation.
LIMITS = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (50, 300),
    "chol": (50, 800),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (30, 250),
    "exang": (0, 1),
    "oldpeak": (-5, 15),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}

LABELS = {
    "age": "Age",
    "sex": "Sex code",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar code",
    "restecg": "Resting ECG code",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina code",
    "oldpeak": "Oldpeak",
    "slope": "ST segment slope code",
    "ca": "Major vessels (ca)",
    "thal": "Thal code",
}

model_bundle = None
model_error = None

try:
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)
    else:
        model_error = "Model not found. Run python train_model.py first."
except Exception:
    model_error = (
        "The saved model could not be loaded. Retrain it in this "
        "environment using python train_model.py."
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template(
            "index.html",
            labels=LABELS,
            error=model_error,
            model_bundle=model_bundle,
        )

    if model_bundle is None:
        return render_template(
            "index.html",
            labels=LABELS,
            error=model_error or "The model is not available.",
            model_bundle=model_bundle,
        ), 503

    values = {}
    errors = []

    for feature in FEATURES:
        raw = request.form.get(feature, "").strip()

        try:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError
        except (ValueError, TypeError):
            errors.append(f"{LABELS[feature]} must be a valid number.")
            continue

        low, high = LIMITS[feature]
        if not low <= value <= high:
            errors.append(
                f"{LABELS[feature]} must be between {low} and {high}."
            )
            continue

        if feature != "oldpeak" and not value.is_integer():
            errors.append(f"{LABELS[feature]} must be a whole number.")
            continue

        values[feature] = value

    if errors:
        return render_template(
            "index.html",
            labels=LABELS,
            error=" ".join(errors),
            form=request.form,
            model_bundle=model_bundle,
        ), 400

    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    pipeline = model_bundle["pipeline"]

    try:
        prediction = int(pipeline.predict(row)[0])
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            labels=LABELS,
            error="Prediction failed. Check the model and input schema.",
            form=request.form,
            model_bundle=model_bundle,
        ), 500

    if prediction == 1:
        result_title = "Model classified the input as target class 1"
        result_message = (
            "The model identified a pattern associated with the positive "
            "class in its training dataset. This does not establish that "
            "the person has heart disease."
        )
    else:
        result_title = "Model classified the input as target class 0"
        result_message = (
            "The model assigned the input to the negative class in its "
            "training dataset. This does not rule out heart disease."
        )

    return render_template(
        "result.html",
        result_title=result_title,
        result_message=result_message,
        prediction=prediction,
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns whether the application is running and whether the model is loaded.
    Does not expose patient information, secrets, or internal details.
    """
    from pathlib import Path
    import joblib

    model_path = Path(__file__).resolve().parent / "models" / "heart_model.joblib"
    model_loaded = model_path.exists()

    if model_loaded:
        try:
            joblib.load(model_path)
            model_status = "loaded"
        except Exception:
            model_status = "error"
    else:
        model_status = "not_loaded"

    status = "ok" if model_loaded else "degraded"
    code = 200 if model_loaded else 503

    return {
        "status": status,
        "model": model_status,
    }, code, {"Content-Type": "application/json"}