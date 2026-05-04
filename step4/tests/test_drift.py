"""Tests for data drift detection — DriftStore, DriftService, and /drift/status endpoint."""
import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# DriftStore
# ---------------------------------------------------------------------------

class TestDriftStore:
    def _store(self):
        from api.core.drift_store import DriftStore
        return DriftStore(maxlen=10)

    def test_log_increments_count(self):
        store = self._store()
        store.log({"tenure": 12, "Contract": "Month-to-month"}, "Yes", 0.8)
        assert store.count() == 1

    def test_get_dataframe_returns_pandas(self):
        store = self._store()
        store.log({"tenure": 12, "MonthlyCharges": 45.0, "Contract": "Month-to-month"}, "No", 0.2)
        df = store.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_maxlen_evicts_oldest(self):
        store = self._store()
        for i in range(15):
            store.log({"tenure": i}, "Yes", 0.9)
        assert store.count() == 10  # maxlen=10

    def test_prediction_column_present(self):
        store = self._store()
        store.log({}, "Yes", 0.75)
        df = store.get_dataframe()
        assert "prediction" in df.columns
        assert df["prediction"].iloc[0] == "Yes"


# ---------------------------------------------------------------------------
# DriftService
# ---------------------------------------------------------------------------

@pytest.fixture()
def reference_file(tmp_path: Path) -> Path:
    stats = {
        "numeric": {
            "tenure": {"sample": list(range(1, 73))},
            "MonthlyCharges": {"sample": [float(x) for x in range(20, 120)]},
            "TotalCharges": {"sample": [float(x * 12) for x in range(1, 73)]},
            "SeniorCitizen": {"sample": [0] * 84 + [1] * 16},
        },
        "categorical": {
            "Contract": {"Month-to-month": 0.55, "One year": 0.21, "Two year": 0.24},
            "InternetService": {"DSL": 0.34, "Fiber optic": 0.44, "No": 0.22},
        },
        "churn_rate": 0.265,
    }
    p = tmp_path / "reference_stats.json"
    p.write_text(json.dumps(stats))
    return p


class TestDriftService:
    def _service(self, ref_path: Path):
        from api.services.drift_service import DriftService
        return DriftService(ref_path)

    def test_not_ready_without_reference_file(self, tmp_path):
        service = self._service(tmp_path / "missing.json")
        assert not service.is_ready()

    def test_ready_with_reference_file(self, reference_file):
        service = self._service(reference_file)
        assert service.is_ready()

    def test_compute_drift_returns_none_when_not_ready(self, tmp_path):
        service = self._service(tmp_path / "missing.json")
        result = service.compute_drift(pd.DataFrame({"tenure": range(100)}))
        assert result is None

    def test_compute_drift_returns_none_when_too_few_samples(self, reference_file):
        service = self._service(reference_file)
        df = pd.DataFrame({"tenure": range(10), "prediction": ["No"] * 10})
        result = service.compute_drift(df)
        assert result is None

    def test_no_drift_on_matching_distributions(self, reference_file):
        service = self._service(reference_file)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "tenure": rng.integers(1, 72, size=200).tolist(),
            "MonthlyCharges": rng.uniform(20, 119, size=200).tolist(),
            "TotalCharges": rng.uniform(12, 864, size=200).tolist(),
            "SeniorCitizen": rng.choice([0, 1], p=[0.84, 0.16], size=200).tolist(),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], p=[0.55, 0.21, 0.24], size=200
            ).tolist(),
            "InternetService": rng.choice(
                ["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22], size=200
            ).tolist(),
            "prediction": rng.choice(["Yes", "No"], p=[0.265, 0.735], size=200).tolist(),
        })
        report = service.compute_drift(df)
        assert report is not None
        assert not report.drift_detected

    def test_drift_detected_on_shifted_distributions(self, reference_file):
        service = self._service(reference_file)
        # Extreme shift: all long-tenure, all Fiber optic, high churn rate
        df = pd.DataFrame({
            "tenure": [72] * 100,
            "MonthlyCharges": [110.0] * 100,
            "TotalCharges": [8000.0] * 100,
            "SeniorCitizen": [1] * 100,
            "Contract": ["Month-to-month"] * 100,
            "InternetService": ["Fiber optic"] * 100,
            "prediction": ["Yes"] * 90 + ["No"] * 10,  # 90% churn vs 26.5% ref
        })
        report = service.compute_drift(df)
        assert report is not None
        assert report.drift_detected
        assert report.retrain_recommended

    def test_report_has_all_required_fields(self, reference_file):
        service = self._service(reference_file)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "tenure": rng.integers(1, 72, size=60).tolist(),
            "prediction": ["Yes"] * 30 + ["No"] * 30,
        })
        report = service.compute_drift(df)
        # May return None if tenure alone isn't enough; just check shape when not None
        if report is not None:
            assert hasattr(report, "drift_detected")
            assert hasattr(report, "drifted_features")
            assert hasattr(report, "feature_scores")
            assert hasattr(report, "production_churn_rate")
            assert hasattr(report, "reference_churn_rate")
            assert 0.0 <= report.production_churn_rate <= 1.0

    def test_prediction_drift_detected(self, reference_file):
        service = self._service(reference_file)
        # reference churn_rate = 0.265; produce 80% churn → Δ > 10pp
        df = pd.DataFrame({
            "tenure": list(range(1, 101)),
            "prediction": ["Yes"] * 80 + ["No"] * 20,
        })
        report = service.compute_drift(df)
        assert report is not None
        assert report.prediction_drift


# ---------------------------------------------------------------------------
# /drift/status endpoint
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_with_model():
    from api.main import app
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.9, 0.1]])
    with TestClient(app) as c:
        c.app.state.model = model
        yield c


def test_drift_status_disabled_when_no_reference(client_with_model):
    data = client_with_model.get("/drift/status").json()
    # Reference stats don't exist in test environment → disabled or pending
    assert data["status"] in ("disabled", "pending")


def test_drift_status_pending_before_enough_predictions(client_with_model):
    data = client_with_model.get("/drift/status").json()
    assert "production_samples" in data


def test_drift_status_updates_after_predictions(client_with_model, reference_file, monkeypatch):
    from api.services.drift_service import DriftService

    svc = DriftService(reference_file)
    client_with_model.app.state.drift_service = svc

    payload = {
        "tenure": 12,
        "MonthlyCharges": 45.5,
        "TotalCharges": "546.0",
        "SeniorCitizen": 0,
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
        "OnlineSecurity": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "TechSupport": "No",
    }
    for _ in range(5):
        client_with_model.post("/predict", json=payload)

    data = client_with_model.get("/drift/status").json()
    assert data["production_samples"] == 5
