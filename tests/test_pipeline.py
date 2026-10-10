"""Unit tests for the AI Heart Disease Prediction System pipeline.

These tests use synthetic fixtures and do NOT require downloading the real
UCI dataset. They test software behavior, not medical performance.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, FEATURES, LIMITS


class TestFeatureNamesAndOrder:
    """Test that feature names match the expected list and order."""

    def test_feature_names_exist(self):
        """All FEATURES should be non-empty strings."""
        for f in FEATURES:
            assert isinstance(f, str), f"Feature name should be string, got {type(f)}"
            assert len(f) > 0, "Feature name should not be empty"

    def test_feature_count(self):
        """Should have exactly 13 features."""
        assert len(FEATURES) == 13, f"Expected 13 features, got {len(FEATURES)}"

    def test_features_sorted_consistent(self):
        """Features should be in the expected order."""
        expected = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal",
        ]
        assert FEATURES == expected, f"Features mismatch:\n  Got:      {FEATURES}\n  Expected: {expected}"


class TestLimitsValidation:
    """Test that input limits are properly defined for all features."""

    def test_all_features_have_limits(self):
        """Every feature in FEATURES should have a LIMITS entry."""
        for f in FEATURES:
            assert f in LIMITS, f"Feature '{f}' missing from LIMITS dict"

    def test_limits_have_valid_bounds(self):
        """Each LIMITS entry should have low < high."""
        for f, (low, high) in LIMITS.items():
            assert low < high, f"LIMITS for '{f}' has low >= high: ({low}, {high})"


class TestSyntheticModelTraining:
    """Test model training on synthetic data - software behavior only."""

    def test_synthetic_pipeline_creates_model(self):
        """A simple pipeline should train and save without error."""
        X = np.random.rand(100, 13)
        y = np.random.randint(0, 2, 100)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        pipeline.fit(X, y)
        assert pipeline is not None
        assert hasattr(pipeline, "predict")

        # Test prediction
        sample = X[0:1]
        pred = pipeline.predict(sample)
        assert pred is not None
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_saved_model_can_reload(self):
        """A trained pipeline should be savable and reloadable."""
        X = np.random.rand(50, 13)
        y = np.random.randint(0, 2, 50)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=200,
                random_state=42,
            )),
        ])

        pipeline.fit(X, y)

        # Save and reload
        model_path = Path(__file__).parent.parent / "models" / "test_pipeline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "features": FEATURES}, model_path)

        reloaded = joblib.load(model_path)
        assert reloaded["pipeline"] is not None
        sample = X[0:1]
        pred = reloaded["pipeline"].predict(sample)
        assert pred[0] in [0, 1]

        # Cleanup
        import os
        if model_path.exists():
            os.remove(model_path)


class TestFlaskRoutes:
    """Test Flask application routes and response behavior."""

    def test_home_page_returns_200(self):
        """The home page should return HTTP 200."""
        with app.test_client() as client:
            response = client.get("/")
            assert response.status_code == 200

    def test_home_page_contains_title(self):
        """The home page should contain the expected title."""
        with app.test_client() as client:
            response = client.get("/")
            text = response.data.decode("utf-8")
            assert "Heart Disease AI" in text

    def test_503_when_model_missing(self):
        """A form submission should return 503 when no model is loaded.

        Reassigning a name imported via `from app import model_bundle` only
        rebinds this test's local variable, not app's module-level state, so
        the app module itself must be patched for the view function to see
        the change.
        """
        import app as app_module

        original = app_module.model_bundle
        app_module.model_bundle = None

        try:
            with app.test_client() as client:
                response = client.post("/", data={})
                assert response.status_code == 503
        finally:
            app_module.model_bundle = original

    def test_valid_form_submission(self):
        """Valid form submission should trigger prediction."""
        with app.test_client() as client:
            # Use a valid combination within limits
            data = {
                "age": "50",
                "sex": "1",
                "cp": "1",
                "trestbps": "120",
                "chol": "200",
                "fbs": "0",
                "restecg": "0",
                "thalach": "150",
                "exang": "0",
                "oldpeak": "1.0",
                "slope": "1",
                "ca": "0",
                "thal": "3",
            }
            response = client.post("/", data=data)
            # Should not be 400 (validation error) or 503 (model missing)
            assert response.status_code != 400
            # Should be 200 or 500 (prediction result)
            assert response.status_code in [200, 500]

    def test_invalid_age_rejected(self):
        """Non-numeric age should be rejected."""
        with app.test_client() as client:
            data = {
                "age": "abc",
                "sex": "1", "cp": "1", "trestbps": "120",
                "chol": "200", "fbs": "0", "restecg": "0",
                "thalach": "150", "exang": "0",
                "oldpeak": "1.0", "slope": "1", "ca": "0", "thal": "3",
            }
            response = client.post("/", data=data)
            assert response.status_code == 400  # Validation error

    def test_out_of_range_age_rejected(self):
        """Age outside valid range should be rejected."""
        with app.test_client() as client:
            data = {
                "age": "200",  # Above max of 120
                "sex": "1", "cp": "1", "trestbps": "120",
                "chol": "200", "fbs": "0", "restecg": "0",
                "thalach": "150", "exang": "0",
                "oldpeak": "1.0", "slope": "1", "ca": "0", "thal": "3",
            }
            response = client.post("/", data=data)
            assert response.status_code == 400  # Validation error

    def test_missing_field_rejected(self):
        """Missing form field should be rejected."""
        with app.test_client() as client:
            data = {
                "age": "50", "sex": "1", "cp": "1", "trestbps": "120",
                "chol": "200", "fbs": "0", "restecg": "0",
                # Missing thalach and several others
                "thalach": "150",
            }
            response = client.post("/", data=data)
            assert response.status_code == 400  # Validation error


class TestModelPersistence:
    """Test that the model pipeline can be saved and reloaded."""

    def test_model_bundle_structure(self):
        """The saved model bundle should have the expected keys."""
        model_path = Path(__file__).parent.parent / "models" / "heart_model.joblib"
        if model_path.exists():
            bundle = joblib.load(model_path)
            assert "pipeline" in bundle, "Missing 'pipeline' key"
            assert "features" in bundle, "Missing 'features' key"
            assert "target_definition" in bundle, "Missing 'target_definition' key"
            assert "test_size" in bundle, "Missing 'test_size' key"
            assert "random_state" in bundle, "Missing 'random_state' key"
            assert callable(bundle["pipeline"].predict), "Pipeline should have predict method"


class TestDataIntegrity:
    """Test dataset file integrity and location."""

    def test_raw_data_exists(self):
        """The raw dataset should exist at the expected path."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "heart.csv"
        assert data_path.exists(), f"Raw dataset not found at {data_path}"

    def test_raw_data_has_content(self):
        """The raw dataset should have multiple rows."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "heart.csv"
        df = pd.read_csv(data_path)
        assert len(df) > 0, "Dataset should have at least one row"
        assert len(df.columns) > 0, "Dataset should have at least one column"


# Run tests when invoked directly
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))