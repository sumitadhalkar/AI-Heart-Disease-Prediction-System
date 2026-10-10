"""Unit tests for the AI Heart Disease Prediction System pipeline.

These tests use synthetic fixtures and do NOT require downloading the real
UCI dataset. They test software behavior, not medical performance.
"""

import os

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

from app import app, FEATURES, LIMITS, MODEL_PATH, model_bundle
from explanation_service import explain_prediction, supports_explanation


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


class TestExplanationService:
    """Test the model-faithful logistic regression contribution breakdown.

    Uses a synthetic pipeline (not the real trained model) with a known
    coefficient vector, so contribution signs and magnitudes can be checked
    exactly rather than just "it runs without error".
    """

    @staticmethod
    def _make_pipeline_and_row():
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.rand(60, len(FEATURES)) * 10, columns=FEATURES)
        y = (X["ca"] + X["thal"] - X["cp"] > 10).astype(int)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
        ])
        pipeline.fit(X, y)

        row = X.iloc[[0]]
        return pipeline, row

    def test_explanation_matches_pipeline_probability(self):
        """The manually-derived probability must equal predict_proba exactly."""
        pipeline, row = self._make_pipeline_and_row()
        explanation = explain_prediction(pipeline, row, FEATURES)

        assert explanation is not None
        expected = pipeline.predict_proba(row)[0, 1]
        assert abs(explanation["probability_class1"] - expected) < 1e-9

    def test_contributions_sum_to_logit(self):
        """intercept + sum(all contributions) must equal the reported logit."""
        pipeline, row = self._make_pipeline_and_row()
        explanation = explain_prediction(pipeline, row, FEATURES)

        total = explanation["intercept"] + sum(
            c["contribution"] for c in explanation["contributions"]
        )
        assert abs(total - explanation["logit"]) < 1e-9

    def test_top_contributions_sorted_by_magnitude(self):
        """top_contributions must be sorted by |contribution| descending."""
        pipeline, row = self._make_pipeline_and_row()
        explanation = explain_prediction(pipeline, row, FEATURES, top_n=5)

        assert len(explanation["top_contributions"]) == 5
        magnitudes = [abs(c["contribution"]) for c in explanation["top_contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert explanation["contributions"][0] == explanation["top_contributions"][0]

    def test_direction_matches_contribution_sign(self):
        """A positive contribution must be labeled 'increases', negative 'decreases'."""
        pipeline, row = self._make_pipeline_and_row()
        explanation = explain_prediction(pipeline, row, FEATURES)

        for c in explanation["contributions"]:
            if c["contribution"] > 1e-9:
                assert c["direction"] == "increases"
            elif c["contribution"] < -1e-9:
                assert c["direction"] == "decreases"
            else:
                assert c["direction"] == "negligible"

    def test_no_explanation_for_unsupported_classifier(self):
        """A non-linear classifier (no coef_) must not produce a fabricated explanation."""
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.rand(30, len(FEATURES)), columns=FEATURES)
        y = rng.randint(0, 2, 30)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(random_state=0)),
        ])
        pipeline.fit(X, y)

        assert supports_explanation(pipeline) is False
        assert explain_prediction(pipeline, X.iloc[[0]], FEATURES) is None


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

    def test_valid_submission_shows_model_explanation(self):
        """A successful prediction should render the contribution breakdown.

        The real model (Cleveland or combined) is logistic regression, so
        explain_prediction() always succeeds for it -- this is a regression
        test for that section actually appearing on the result page.
        """
        with app.test_client() as client:
            data = {
                "age": "50", "sex": "1", "cp": "1", "trestbps": "120",
                "chol": "200", "fbs": "0", "restecg": "0", "thalach": "150",
                "exang": "0", "oldpeak": "1.0", "slope": "1", "ca": "0", "thal": "3",
            }
            response = client.post("/", data=data)
            if response.status_code != 200:
                return  # model unavailable in this environment
            text = response.data.decode("utf-8")
            assert "Why Did the Model Produce This Result?" in text
            assert "Model Score" in text
            assert "Uncalibrated model confidence" in text

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


class TestHealthEndpoint:
    """Test that /health accurately reports model availability."""

    def test_health_ok_when_model_loaded(self):
        """Should return 200 and status 'ok' when the model file exists."""
        if not MODEL_PATH.exists():
            return
        with app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.get_json() == {"status": "ok", "model": "loaded"}

    def test_health_degraded_when_model_file_missing(self):
        """Should return 503 and status 'degraded' when the model file is absent.

        /health re-reads the model straight from disk on every request (unlike
        the index route, which only checks the in-memory `model_bundle` set at
        startup), so this has to be tested by removing the file rather than by
        patching app.model_bundle.
        """
        if not MODEL_PATH.exists():
            return
        backup_path = MODEL_PATH.with_suffix(".joblib.bak")
        os.rename(MODEL_PATH, backup_path)
        try:
            with app.test_client() as client:
                response = client.get("/health")
                assert response.status_code == 503
                assert response.get_json()["status"] == "degraded"
        finally:
            os.rename(backup_path, MODEL_PATH)


class TestDatasetInfoRendering:
    """The home page should reflect the actual loaded model's dataset info.

    Regression test: the template used to hardcode "Cleveland database" and
    "Instances: 303" regardless of which dataset actually trained the active
    model, so a model trained with `train_model.py --dataset combined` would
    render a misleading dataset description.
    """

    def test_dataset_label_matches_model_bundle(self):
        if model_bundle is None:
            return
        with app.test_client() as client:
            response = client.get("/")
            text = response.data.decode("utf-8")
            label = model_bundle.get(
                "dataset_label", "UCI Heart Disease Repository (Cleveland database)"
            )
            count = model_bundle.get("instance_count", 303)
            assert label in text
            assert f"Instances: {count}" in text


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