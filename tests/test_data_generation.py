"""
Tests for Synthetic EHR and Time-Series Data Generation.
"""

import pandas as pd
import pytest

from src.data.generate_ehr import generate_ehr_cohort
from src.data.generate_timeseries import generate_patient_timeseries


def test_generate_ehr_cohort():
    """Validates synthetic EHR generation properties and clinical boundaries."""
    df_ehr = generate_ehr_cohort(num_patients=20, seed=123)
    assert len(df_ehr) == 20
    assert "patient_id" in df_ehr.columns
    assert "bmi" in df_ehr.columns
    assert "hba1c" in df_ehr.columns
    assert "fasting_glucose" in df_ehr.columns

    # Check plausible medical ranges
    assert df_ehr["age"].between(30, 90).all()
    assert df_ehr["bmi"].between(18.0, 50.0).all()
    assert df_ehr["hba1c"].between(6.0, 13.0).all()
    assert df_ehr["fasting_glucose"].between(90, 300).all()


def test_generate_patient_timeseries():
    """Validates multimodal time-series generator continuity and bounds."""
    patient_row = {
        "patient_id": "PT-SYNTH-TEST",
        "name": "Test Patient",
        "bmi": 28.5,
        "hba1c": 7.5,
        "fasting_glucose": 130.0,
    }
    df_stream = generate_patient_timeseries(patient_row, days=3, seed=42)

    # 3 days * 96 ticks/day = 288 readings
    assert len(df_stream) == 288
    assert "glucose" in df_stream.columns
    assert "heart_rate" in df_stream.columns
    assert "steps" in df_stream.columns

    # Verify physiological ranges
    assert df_stream["glucose"].between(40.0, 450.0).all()
    assert df_stream["heart_rate"].between(45.0, 180.0).all()
    assert (df_stream["steps"] >= 0).all()
