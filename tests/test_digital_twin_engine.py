"""
Tests for Digital Twin Ingestion and State Management Engine.
"""

import pytest

from src.digital_twin.engine import DigitalTwinEngine
from src.digital_twin.ingestion import SensorValidationError, validate_sensor_reading


def test_validate_sensor_reading():
    """Tests boundary validation and error rejection."""
    # Valid reading
    valid_data = {"glucose": 135.5, "heart_rate": 74.0, "steps": 250}
    sanitized, warnings = validate_sensor_reading(valid_data)
    assert sanitized["glucose"] == 135.5
    assert sanitized["heart_rate"] == 74.0
    assert sanitized["steps"] == 250

    # Physiologically impossible glucose
    with pytest.raises(SensorValidationError):
        validate_sensor_reading({"glucose": 12.0})  # < 20 mg/dL

    with pytest.raises(SensorValidationError):
        validate_sensor_reading({"glucose": 750.0})  # > 600 mg/dL

    # Physiologically impossible HR
    with pytest.raises(SensorValidationError):
        validate_sensor_reading({"glucose": 120.0, "heart_rate": 15.0})


def test_digital_twin_engine_lifecycle():
    """Tests the full live loop: ingestion -> state update -> risk forecast."""
    patient = {
        "patient_id": "PT-TEST-01",
        "name": "Test Twin Patient",
        "age": 60,
        "bmi": 30.0,
        "hba1c": 8.0,
        "fasting_glucose": 140.0,
    }
    engine = DigitalTwinEngine(patient_id="PT-TEST-01", ehr_profile=patient)

    # Ingest 10 consecutive ticks
    for i in range(10):
        tick = {
            "timestamp": f"2026-06-01 10:{i*15:02d}:00",
            "glucose": 120.0 + i * 5,
            "heart_rate": 72.0 + i,
            "steps": 50,
            "sleep_hours": 6.5,
            "sleep_quality": 68.0,
        }
        res = engine.process_observation(tick)

    # Verify state updated
    assert len(engine.state.telemetry_buffer) == 10
    assert engine.state.derived_state["glucose_velocity_mgdl_per_min"] > 0.0

    pred = engine.state.latest_prediction
    assert 0.0 <= pred["spike_probability_2h"] <= 1.0
    assert pred["risk_level"] in ["Low", "Moderate", "High"]
    assert len(pred["predicted_trajectory"]) == 8
