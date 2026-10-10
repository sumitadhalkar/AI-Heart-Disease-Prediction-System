#!/usr/bin/env python3
"""
Sample prediction script for the AI Heart Disease Prediction System.

Loads the actual trained model pipeline and runs a sample inference
using the exact feature names, order, and encodings expected by the model.

This is an educational prototype - predictions are NOT medical diagnoses.
"""

from pathlib import Path

import numpy as np
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]

# A documented sample input using the exact expected feature names,
# order, and dataset encodings. All values are valid within the
# project's input LIMITS and match the training dataset's coding.
SAMPLE_INPUT = {
    "age": 52,                    # age in years (1-120, whole number)
    "sex": 1,                     # 1 = Male in dataset
    "cp": 1,                      # 1 = Typical angina
    "trestbps": 130,              # Resting blood pressure (mm Hg), range 50-300
    "chol": 235,                  # Serum cholesterol (mg/dL), range 50-800
    "fbs": 0,                     # 0 = At or below 120 mg/dL
    "restecg": 0,                 # 0 = Normal
    "thalach": 150,               # Max heart rate achieved, range 30-250
    "exang": 0,                   # 0 = No exercise-induced angina
    "oldpeak": 1.0,               # ST depression, step-any allowed, range -5 to 15
    "slope": 1,                   # 1 = Upsloping
    "ca": 0,                      # 0 major vessels colored by fluoroscopy
    "thal": 3,                    # 3 = Normal
}


def validate_sample_input(input_dict, features):
    """Validate that the sample input contains all required features with valid values."""
    errors = []

    for feature in features:
        if feature not in input_dict:
            errors.append(f"Missing feature: {feature}")
            continue

        value = input_dict[feature]

        # Check numeric
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            errors.append(f"{feature} must be a valid number.")
            continue

        # Check finite
        if not np.isfinite(float_val):
            errors.append(f"{feature} must be a finite number.")
            continue

        # Check range limits
        from app import LIMITS
        if feature in LIMITS:
            low, high = LIMITS[feature]
            if not (low <= float_val <= high):
                errors.append(
                    f"{feature} must be between {low} and {high} "
                    f"(got {float_val})."
                )

        # Check whole number for non-oldpeak features
        if feature != "oldpeak" and not float_val.is_integer():
            errors.append(
                f"{feature} must be a whole number "
                f"(got {float_val})."
            )

    return errors


def main():
    print("=" * 60)
    print("AI HEART DISEASE PREDICTION SYSTEM")
    print("=" * 60)
    print()

    # Load the trained model pipeline
    model_loaded = False
    if MODEL_PATH.exists():
        try:
            model_bundle = joblib.load(MODEL_PATH)
            pipeline = model_bundle["pipeline"]
            model_loaded = True
            print("Model loaded: SUCCESS")
        except Exception:
            print("Model loaded: FAILED - could not load pipeline.")
            print(f"Model path: {MODEL_PATH}")
    else:
        print("Model loaded: FAILED - model file not found.")
        print(f"Model path: {MODEL_PATH}")
        print()
        print("Run 'python train_model.py' to train and save the model.")
        print()
        return

    # Validate the sample input
    print("Input validation:")
    validation_errors = validate_sample_input(SAMPLE_INPUT, FEATURES)
    if validation_errors:
        for error in validation_errors:
            print(f"  ERROR: {error}")
        print()
        print("Please fix the validation errors above.")
        return
    else:
        print("  Input validation: SUCCESS")
    print()

    # Run the model's actual prediction pipeline
    row = pd.DataFrame([[SAMPLE_INPUT[f] for f in FEATURES]], columns=FEATURES)

    try:
        prediction = int(pipeline.predict(row)[0])
    except Exception:
        print("Prediction FAILED - check the model and input schema.")
        print()
        return

    # Display results
    print("Prediction:")
    print(f"  Output class: {prediction}")
    print()
    if prediction == 1:
        print("  The model identified a pattern associated with the positive")
        print("  class in its training dataset.")
    else:
        print("  The model assigned the input to the negative class in its")
        print("  training dataset.")
    print()
    print("Status: Sample inference completed successfully.")
    print()
    print("Note: This is an illustrative software test,")
    print("      not a medical diagnosis.")
    print()
    print("The output class above is only an illustration of formatting.")
    print("The displayed output class is not an individual's validated")
    print("probability of having heart disease.")
    print()
    print("=" * 60)
    print("End of prediction output")
    print("=" * 60)


if __name__ == "__main__":
    main()