"""
Pydantic Schemas for Healthcare Digital Twin REST APIs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PatientProfileResponse(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    bmi: float
    diagnosis: str
    diabetes_duration_years: float
    hba1c: float
    fasting_glucose: float
    systolic_bp: int
    diastolic_bp: int
    medications_display: str
    comorbidities: str
    family_history: int


class SensorReadingInput(BaseModel):
    patient_id: str = Field(default="PT-SYNTH-001", description="Target patient ID")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp (YYYY-MM-DD HH:MM:SS)")
    glucose: float = Field(..., description="Continuous Glucose Monitoring reading (mg/dL)")
    heart_rate: Optional[float] = Field(default=72.0, description="Optical heart rate (bpm)")
    steps: Optional[int] = Field(default=0, description="Step count in 15-minute interval")
    sleep_hours: Optional[float] = Field(default=7.0, description="Sleep duration from previous night")
    sleep_quality: Optional[float] = Field(default=70.0, description="Sleep quality score (0-100)")
    carbs_intake: Optional[float] = Field(default=0.0, description="Carbohydrates consumed in grams")
    activity_type: Optional[str] = Field(default="sedentary", description="Activity classification")


class SensorReadingResponse(BaseModel):
    status: str
    validated_reading: Dict[str, Any]
    validation_warnings: List[str]


class RiskFactorItem(BaseModel):
    feature: str
    label: str
    contribution: str
    attribution_value: float
    direction: str
    current_value: float


class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: Optional[str]
    spike_probability_2h: float
    risk_level: str
    predicted_trajectory: List[float]
    trajectory_timestamps: List[str]
    confidence_lower: List[float]
    confidence_upper: List[float]
    primary_risk_factors: List[RiskFactorItem]


class DerivedState(BaseModel):
    glucose_velocity_mgdl_per_min: float
    glucose_delta_30m: float
    glucose_mean_2h: float
    glucose_std_2h: float
    cv_percent: float
    autonomic_hr_elevation: float
    active_steps_2h: int


class DigitalTwinStateResponse(BaseModel):
    patient_id: str
    ehr_profile: Dict[str, Any]
    latest_observation: Optional[Dict[str, Any]]
    derived_state: DerivedState
    prediction: Dict[str, Any]
    buffer_length: int


class SimulateRequest(BaseModel):
    scenario: str = Field(
        default="postprandial_spike",
        description="Scenario key: 'postprandial_spike', 'euglycemic_walk', or 'dawn_phenomenon'"
    )
    num_steps: int = Field(default=5, ge=1, le=24, description="Number of 15-minute steps to advance")


class SimulateStepResult(BaseModel):
    step: int
    timestamp: str
    glucose: float
    heart_rate: float
    steps: int
    spike_probability_2h: float
    risk_level: str
    glucose_velocity: float


class SimulateResponse(BaseModel):
    scenario: str
    steps_executed: int
    results: List[SimulateStepResult]
    final_state: Dict[str, Any]
