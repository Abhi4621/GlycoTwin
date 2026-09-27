"""
Tests for FastAPI Backend Endpoints using TestClient.
"""

from fastapi.testclient import TestClient
import pytest

from backend.app import app

client = TestClient(app)


def test_root_endpoint():
    """Validates root health check and disclaimer."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "disclaimer" in data


def test_get_patient_profile():
    """Validates patient EHR retrieval."""
    resp = client.get("/patient/PT-SYNTH-001")
    assert resp.status_code == 200
    profile = resp.json()
    assert profile["patient_id"] == "PT-SYNTH-001"
    assert "diagnosis" in profile
    assert profile["hba1c"] > 0


def test_ingest_sensor_data():
    """Validates /sensor-data POST endpoint and state update."""
    payload = {
        "patient_id": "PT-SYNTH-001",
        "timestamp": "2026-06-01 12:00:00",
        "glucose": 145.0,
        "heart_rate": 78.0,
        "steps": 120,
        "sleep_hours": 7.0,
        "sleep_quality": 80.0,
        "carbs_intake": 45.0,
        "activity_type": "light_active",
    }
    resp = client.post("/sensor-data", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["validated_reading"]["glucose"] == 145.0


def test_get_prediction_and_risk_factors():
    """Validates /prediction and /risk-factors endpoints."""
    resp_pred = client.get("/patient/PT-SYNTH-001/prediction")
    assert resp_pred.status_code == 200
    pred = resp_pred.json()
    assert 0.0 <= pred["spike_probability_2h"] <= 1.0
    assert pred["risk_level"] in ["Low", "Moderate", "High"]

    resp_rf = client.get("/patient/PT-SYNTH-001/risk-factors")
    assert resp_rf.status_code == 200
    rf_data = resp_rf.json()
    assert "primary_risk_factors" in rf_data


def test_simulate_endpoint():
    """Validates /simulate POST endpoint."""
    sim_payload = {"scenario": "postprandial_spike", "num_steps": 3}
    resp = client.post("/simulate", json=sim_payload)
    assert resp.status_code == 200
    sim_data = resp.json()
    assert sim_data["steps_executed"] == 3
    assert len(sim_data["results"]) == 3
