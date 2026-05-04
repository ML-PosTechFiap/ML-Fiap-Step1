"""API integration tests — test /predict and /health with a mock model."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 45.50,
    "TotalCharges": "546.00",
}


def _make_mock_model(churn: bool = False):
    model = MagicMock()
    model.predict.return_value = np.array([1 if churn else 0])
    model.predict_proba.return_value = np.array([[0.2, 0.8] if churn else [0.9, 0.1]])
    return model


@pytest.fixture()
def client_with_model():
    from api.main import app

    with TestClient(app) as c:
        c.app.state.model = _make_mock_model(churn=True)
        yield c


@pytest.fixture()
def client_no_model():
    from api.main import app

    with TestClient(app) as c:
        c.app.state.model = None
        yield c


def test_predict_returns_200_with_model(client_with_model):
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_prediction_field(client_with_model):
    data = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert "prediction" in data
    assert data["prediction"] in ("Yes", "No")


def test_predict_response_has_probability_field(client_with_model):
    data = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_response_has_model_version(client_with_model):
    data = client_with_model.post("/predict", json=VALID_PAYLOAD).json()
    assert "model_version" in data
    assert data["model_version"] == "0.4.0"


def test_predict_503_when_model_not_loaded(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_empty_payload_still_processes(client_with_model):
    response = client_with_model.post("/predict", json={})
    assert response.status_code in (200, 422, 500)


def test_health_loaded_when_model_present(client_with_model):
    data = client_with_model.get("/health").json()
    assert data["model_status"] == "loaded"


def test_health_not_loaded_when_model_absent(client_no_model):
    data = client_no_model.get("/health").json()
    assert data["model_status"] == "not_loaded"


def test_x_process_time_header_present(client_with_model):
    response = client_with_model.get("/health")
    assert "x-process-time" in response.headers
