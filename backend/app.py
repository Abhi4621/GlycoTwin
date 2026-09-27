"""
FastAPI Backend Application for Type 2 Diabetes Healthcare Digital Twin.
Implements RESTful endpoints for clinical profiles, sensor ingestion,
real-time Digital Twin state tracking, risk forecasts, and stream simulations.
"""

import datetime
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.database import (
    get_patient,
    get_patient_history,
    init_db,
    insert_sensor_reading,
    log_prediction,
    save_patient,
)
from backend.schemas import (
    DigitalTwinStateResponse,
    PatientProfileResponse,
    PredictionResponse,
    RiskFactorItem,
    SensorReadingInput,
    SensorReadingResponse,
    SimulateRequest,
    SimulateResponse,
    SimulateStepResult,
)
from src.data.loader import load_patient_data
from src.digital_twin.engine import DigitalTwinEngine
from src.digital_twin.ingestion import SensorValidationError


# Initialize FastAPI App
app = FastAPI(
    title="Type 2 Diabetes Digital Twin API",
    description=(
        "RESTful API maintaining dynamic physiological Digital Twins for Type 2 Diabetes patients. "
        "Forecasts short-term glucose spikes (< 2 hours) and surfaces clinical risk factors. "
        "Educational and research proof-of-concept prototype (Happiest Health Challenge 2026)."
    ),
    version="1.0.0",
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory repository of active Digital Twin engines keyed by patient_id
ACTIVE_ENGINES: Dict[str, DigitalTwinEngine] = {}


def get_or_create_engine(patient_id: str) -> DigitalTwinEngine:
    """Retrieves an active Digital Twin engine or instantiates and seeds a new one."""
    if patient_id not in ACTIVE_ENGINES:
        init_db()
        ehr_dict = get_patient(patient_id)
        if not ehr_dict:
            # Load default patient and seed database
            ehr_profile, history_stream = load_patient_data(patient_id=patient_id)
            save_patient(ehr_profile)
            engine = DigitalTwinEngine(patient_id=patient_id, ehr_profile=ehr_profile)
            engine.seed_initial_history(history_stream)
        else:
            engine = DigitalTwinEngine(patient_id=patient_id, ehr_profile=ehr_dict)
            history = get_patient_history(patient_id, limit=64)
            if history:
                import pandas as pd
                engine.seed_initial_history(pd.DataFrame(history))
            else:
                _, history_stream = load_patient_data(patient_id=patient_id)
                engine.seed_initial_history(history_stream)

        ACTIVE_ENGINES[patient_id] = engine

    return ACTIVE_ENGINES[patient_id]


@app.on_event("startup")
def startup_event():
    """Initializes the database and pre-warms the default patient Digital Twin."""
    init_db()
    # Pre-warm default demonstration patient
    get_or_create_engine("PT-SYNTH-001")


@app.get("/", tags=["System"])
def root():
    """Root health check and disclaimer declaration."""
    return {
        "system": "Healthcare Digital Twin - Type 2 Diabetes Short-Term Glucose Spike Forecaster",
        "competition": "Happiest Health Digital Twin Challenge 2026",
        "version": "1.0.0",
        "status": "operational",
        "disclaimer": (
            "NOTICE: Educational and research proof-of-concept prototype. "
            "All data is synthetic. Not a diagnostic tool. Do not use for clinical medication decisions."
        ),
        "endpoints": [
            "GET /patient/{id}",
            "GET /patient/{id}/history",
            "POST /sensor-data",
            "GET /patient/{id}/current-state",
            "GET /patient/{id}/prediction",
            "GET /patient/{id}/risk-factors",
            "POST /simulate",
        ],
    }


@app.get("/patient/{patient_id}", response_model=PatientProfileResponse, tags=["Patient"])
def get_patient_profile(patient_id: str):
    """Returns static demographic and clinical EHR profile for the specified patient."""
    engine = get_or_create_engine(patient_id)
    profile = engine.ehr_profile
    return PatientProfileResponse(
        patient_id=engine.patient_id,
        name=profile.get("name", "Arthur Pendelton"),
        age=int(profile.get("age", 58)),
        gender=profile.get("gender", "Male"),
        bmi=float(profile.get("bmi", 29.5)),
        diagnosis=profile.get("diagnosis", "Type 2 Diabetes Mellitus"),
        diabetes_duration_years=float(profile.get("diabetes_duration_years", 6.0)),
        hba1c=float(profile.get("hba1c", 8.1)),
        fasting_glucose=float(profile.get("fasting_glucose", 138.0)),
        systolic_bp=int(profile.get("systolic_bp", 134)),
        diastolic_bp=int(profile.get("diastolic_bp", 86)),
        medications_display=profile.get("medications_display", "Metformin 1000mg BID"),
        comorbidities=profile.get("comorbidities", "Essential Hypertension, Dyslipidemia"),
        family_history=int(profile.get("family_history_diabetes", 1)),
    )


@app.get("/patient/{patient_id}/history", tags=["Telemetry"])
def get_history(patient_id: str, limit: int = 96):
    """Returns recent continuous sensor telemetry readings for the patient."""
    engine = get_or_create_engine(patient_id)
    buf = list(engine.state.telemetry_buffer)
    if buf:
        return {"patient_id": patient_id, "readings_count": len(buf), "history": buf[-limit:]}
    history = get_patient_history(patient_id, limit=limit)
    return {"patient_id": patient_id, "readings_count": len(history), "history": history}


@app.post("/sensor-data", response_model=SensorReadingResponse, tags=["Digital Twin State"])
def ingest_sensor_data(reading: SensorReadingInput):
    """
    Ingests an incoming sensor reading into the patient's Digital Twin.
    Executes the continuous twin update cycle:
    Validation -> State Update -> Feature Extraction -> ML Spike Inference ->
    Trajectory Projection -> SHAP Explanation -> Database Persistence.
    """
    engine = get_or_create_engine(reading.patient_id)
    raw_dict = reading.dict()

    try:
        updated_state = engine.process_observation(raw_dict)
    except SensorValidationError as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err))

    # Persist validated reading to DB
    insert_sensor_reading(engine.state.latest_observation)

    # Persist prediction log
    pred = engine.state.latest_prediction
    log_prediction(
        patient_id=reading.patient_id,
        timestamp=engine.state.latest_observation.get("timestamp", datetime.datetime.now().isoformat()),
        spike_prob=pred.get("spike_probability_2h", 0.0),
        risk_level=pred.get("risk_level", "Low"),
        trajectory=pred.get("predicted_trajectory", []),
        risk_factors=pred.get("risk_factors", []),
    )

    return SensorReadingResponse(
        status="success",
        validated_reading=engine.state.latest_observation,
        validation_warnings=updated_state.get("validation_warnings", []),
    )


@app.get("/patient/{patient_id}/current-state", response_model=DigitalTwinStateResponse, tags=["Digital Twin State"])
def get_current_twin_state(patient_id: str):
    """Returns the comprehensive active physiological Digital Twin state."""
    engine = get_or_create_engine(patient_id)
    state_dict = engine.state.to_dict()
    return DigitalTwinStateResponse(
        patient_id=engine.patient_id,
        ehr_profile=state_dict["ehr_profile"],
        latest_observation=state_dict["latest_observation"],
        derived_state=state_dict["derived_state"],
        prediction=state_dict["prediction"],
        buffer_length=state_dict["buffer_length"],
    )


@app.get("/patient/{patient_id}/prediction", response_model=PredictionResponse, tags=["Prediction"])
def get_patient_prediction(patient_id: str):
    """Returns the latest short-term (2-hour) glucose spike risk and forecast trajectory."""
    engine = get_or_create_engine(patient_id)
    pred = engine.state.latest_prediction
    obs = engine.state.latest_observation
    ts = obs.get("timestamp") if obs else None

    factors = [
        RiskFactorItem(
            feature=f.get("feature", "unknown"),
            label=f.get("label", "Unknown Factor"),
            contribution=f.get("contribution", "+0%"),
            attribution_value=f.get("attribution_value", 0.0),
            direction=f.get("direction", "increases_risk"),
            current_value=f.get("current_value", 0.0),
        )
        for f in pred.get("risk_factors", [])
    ]

    return PredictionResponse(
        patient_id=engine.patient_id,
        timestamp=ts,
        spike_probability_2h=pred.get("spike_probability_2h", 0.15),
        risk_level=pred.get("risk_level", "Low"),
        predicted_trajectory=pred.get("predicted_trajectory", []),
        trajectory_timestamps=pred.get("trajectory_timestamps", []),
        confidence_lower=pred.get("confidence_lower", []),
        confidence_upper=pred.get("confidence_upper", []),
        primary_risk_factors=factors,
    )


@app.get("/patient/{patient_id}/risk-factors", tags=["Prediction"])
def get_risk_factors(patient_id: str):
    """Returns the SHAP-derived physiological risk factors driving the current prediction."""
    engine = get_or_create_engine(patient_id)
    return {
        "patient_id": engine.patient_id,
        "risk_level": engine.state.latest_prediction.get("risk_level", "Low"),
        "spike_probability_2h": engine.state.latest_prediction.get("spike_probability_2h", 0.15),
        "primary_risk_factors": engine.state.latest_prediction.get("risk_factors", []),
    }


@app.post("/simulate", response_model=SimulateResponse, tags=["Simulation"])
def simulate_scenario(req: SimulateRequest):
    """
    Executes a discrete sequence of simulated wearable observations for the patient twin.
    Demonstrates dynamic transitions (e.g., normal baseline -> postprandial spike -> high risk).
    """
    from simulation.scenarios import get_scenario_ticks

    patient_id = "PT-SYNTH-001"
    engine = get_or_create_engine(patient_id)

    ticks = get_scenario_ticks(scenario=req.scenario, num_steps=req.num_steps)
    step_results: List[SimulateStepResult] = []

    for i, tick in enumerate(ticks, start=1):
        tick["patient_id"] = patient_id
        engine.process_observation(tick)

        pred = engine.state.latest_prediction
        obs = engine.state.latest_observation
        derived = engine.state.derived_state

        step_results.append(
            SimulateStepResult(
                step=i,
                timestamp=obs.get("timestamp", ""),
                glucose=obs.get("glucose", 0.0),
                heart_rate=obs.get("heart_rate", 72.0),
                steps=obs.get("steps", 0),
                spike_probability_2h=pred.get("spike_probability_2h", 0.0),
                risk_level=pred.get("risk_level", "Low"),
                glucose_velocity=derived.get("glucose_velocity_mgdl_per_min", 0.0),
            )
        )

    return SimulateResponse(
        scenario=req.scenario,
        steps_executed=len(step_results),
        results=step_results,
        final_state=engine.state.to_dict(),
    )
