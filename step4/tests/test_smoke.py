"""Smoke tests — verify the app boots and core routes respond."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api.main import app

    with TestClient(app) as c:
        c.app.state.model = None
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body_has_status_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_reports_model_not_loaded_when_absent(client):
    data = client.get("/health").json()
    assert data["model_status"] == "not_loaded"


def test_metrics_endpoint_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_docs_endpoint_available(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_reachable(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "ML-Churn-Prediction"
