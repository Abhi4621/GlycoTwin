"""
Multimodal Physiological Time-Series Generator for Type 2 Diabetes Digital Twin.
Simulates coupled dynamics between Continuous Glucose Monitoring (CGM), Heart Rate,
Activity Steps, Sleep Architecture, and Carbohydrate Ingestion at 15-minute intervals.
Purely synthetic for research and education.
"""

import argparse
import datetime
import os
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def simulate_glucose_curve(
    num_ticks: int,
    base_glucose: float,
    insulin_sensitivity: float,
    meals: np.ndarray,
    steps: np.ndarray,
    sleep_factors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulates continuous glucose time-series using a discrete compartmental model
    capturing carbohydrate absorption, exercise utilization, and circadian dawn phenomenon.
    """
    glucose = np.zeros(num_ticks)
    # Start at patient's baseline fasting glucose
    current_g = base_glucose

    # Meal absorption buffer: two-compartment gut absorption
    gut_active_carbs = np.zeros(num_ticks)
    for t in range(num_ticks):
        if meals[t] > 0:
            # Spread carb appearance over the next 12 ticks (3 hours)
            # Peak appearance around tick 3-4 (45-60 min)
            absorption_window = min(12, num_ticks - t)
            shape = np.array([0.1, 0.25, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005][:absorption_window])
            shape = shape / np.sum(shape)
            gut_active_carbs[t:t + absorption_window] += meals[t] * shape

    # Autoregressive simulation
    for t in range(num_ticks):
        # Time of day in hours (0 to 24)
        hour = (t % 96) * 0.25

        # Circadian baseline fluctuation (dawn phenomenon ~05:00 - 08:30)
        dawn_surge = 14.0 * np.exp(-((hour - 6.5) ** 2) / (2 * 1.5 ** 2))
        circadian_drift = base_glucose + dawn_surge - 5.0 * np.sin(2 * np.pi * hour / 24)

        # Carb influx scaled by patient insulin sensitivity and sleep resistance
        # Lower insulin sensitivity (higher resistance) means greater spike
        carb_impact = gut_active_carbs[t] * (2.4 / max(insulin_sensitivity, 0.2)) * sleep_factors[t]

        # Exercise glucose uptake (GLUT4 translocation from muscle contraction)
        exercise_clearance = (steps[t] / 400.0) * 4.2

        # Homeostatic return towards circadian baseline
        homeostasis_pull = 0.08 * (circadian_drift - current_g)

        # Autonomic random shock
        noise = rng.normal(0, 1.2)

        # Next state
        delta_g = homeostasis_pull + carb_impact - exercise_clearance + noise
        current_g = np.clip(current_g + delta_g, 55.0, 380.0)
        glucose[t] = float(np.round(current_g, 1))

    return glucose


def generate_patient_timeseries(
    patient_row: Union[pd.Series, Dict],
    days: int = 60,
    start_date: str = "2026-06-01 00:00:00",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic 15-minute multimodal telemetry for a specific patient.

    Parameters:
        patient_row: Dict or Series containing patient clinical EHR attributes.
        days: Number of days to simulate (96 ticks per day).
        start_date: Starting ISO timestamp string.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with 15-minute sensor measurements and clinical event markers.
    """
    rng = np.random.default_rng(seed)
    num_ticks = days * 96
    start_dt = datetime.datetime.fromisoformat(start_date)
    timestamps = [start_dt + datetime.timedelta(minutes=15 * i) for i in range(num_ticks)]

    # Clinical characteristics
    base_fasting = float(patient_row.get("fasting_glucose", 135.0))
    bmi = float(patient_row.get("bmi", 29.5))
    hba1c = float(patient_row.get("hba1c", 7.8))
    patient_id = patient_row.get("patient_id", "PT-SYNTH-001")

    # Insulin sensitivity metric (scaled 0.3 to 1.5, lower in high BMI/HbA1c)
    insulin_sensitivity = float(np.clip(1.8 - 0.03 * (bmi - 22.0) - 0.09 * (hba1c - 6.0), 0.35, 1.4))

    # Pre-generate daily sleep metrics
    daily_sleep_hours = np.clip(rng.normal(6.8, 1.1, size=days), 4.2, 9.2)
    daily_sleep_quality = np.clip(rng.normal(72, 14, size=days), 35, 96)
    # Sleep resistance factor: poor sleep increases morning/daytime insulin resistance
    daily_sleep_resistance = 1.0 + np.where(daily_sleep_hours < 6.0, 0.22, 0.0) + np.where(daily_sleep_quality < 60, 0.15, 0.0)

    sleep_factors = np.repeat(daily_sleep_resistance, 96)
    sleep_hours_series = np.repeat(daily_sleep_hours, 96)
    sleep_quality_series = np.repeat(daily_sleep_quality, 96)

    # Generate daily lifestyle events (meals and physical activity)
    meals = np.zeros(num_ticks)
    steps = np.zeros(num_ticks, dtype=int)
    activity_types = ["sedentary"] * num_ticks
    heart_rates = np.zeros(num_ticks)

    baseline_resting_hr = 66.0 + 0.5 * (bmi - 25.0)

    for day_idx in range(days):
        day_offset = day_idx * 96

        # Sleep period: 23:00 to 07:00 (tick 92 to 28 next morning approx)
        for tick in range(day_offset, day_offset + 96):
            hour = (tick % 96) * 0.25
            if hour >= 23.0 or hour < 6.5:
                activity_types[tick] = "sleeping"
                steps[tick] = int(rng.poisson(3))
            elif 6.5 <= hour < 8.5:
                # Morning routine
                steps[tick] = int(rng.poisson(180))
                activity_types[tick] = "light_active"
            elif 8.5 <= hour < 12.0:
                # Work / sedentary morning
                steps[tick] = int(rng.poisson(80))
                activity_types[tick] = "sedentary"
            elif 12.0 <= hour < 13.5:
                # Lunch break walk
                steps[tick] = int(rng.poisson(280))
                activity_types[tick] = "light_active"
            elif 13.5 <= hour < 17.5:
                # Afternoon sedentary
                steps[tick] = int(rng.poisson(75))
                activity_types[tick] = "sedentary"
            elif 17.5 <= hour < 19.0:
                # Evening exercise or commute
                if rng.uniform() < 0.45:
                    steps[tick] = int(rng.poisson(1200))
                    activity_types[tick] = "moderate_walk"
                else:
                    steps[tick] = int(rng.poisson(250))
                    activity_types[tick] = "light_active"
            else:
                # Evening dinner and relaxing
                steps[tick] = int(rng.poisson(90))
                activity_types[tick] = "sedentary"

        # Scheduled Meals (with slight timing jitter)
        # Breakfast: ~08:00 (tick 32)
        b_tick = day_offset + 32 + int(rng.integers(-2, 3))
        b_carbs = rng.normal(45, 12)
        meals[b_tick] = max(15.0, b_carbs)

        # Lunch: ~12:45 (tick 51)
        l_tick = day_offset + 51 + int(rng.integers(-3, 3))
        l_carbs = rng.normal(65, 18)
        meals[l_tick] = max(25.0, l_carbs)

        # Dinner: ~19:30 (tick 78)
        d_tick = day_offset + 78 + int(rng.integers(-3, 4))
        d_carbs = rng.normal(70, 20)
        meals[d_tick] = max(30.0, d_carbs)

        # Occasional afternoon snack: ~16:00 (tick 64)
        if rng.uniform() < 0.40:
            s_tick = day_offset + 64 + int(rng.integers(-2, 3))
            meals[s_tick] = rng.uniform(15, 35)

    # Simulate physiological glucose curve
    glucose = simulate_glucose_curve(
        num_ticks=num_ticks,
        base_glucose=base_fasting,
        insulin_sensitivity=insulin_sensitivity,
        meals=meals,
        steps=steps,
        sleep_factors=sleep_factors,
        rng=rng,
    )

    # Simulate dynamic heart rate linked to activity, sleep, and meals
    for t in range(num_ticks):
        act = activity_types[t]
        hour = (t % 96) * 0.25

        if act == "sleeping":
            base_hr = baseline_resting_hr - 8.0 + 2.0 * np.sin(2 * np.pi * hour / 24)
        elif act == "sedentary":
            base_hr = baseline_resting_hr + 2.0
        elif act == "light_active":
            base_hr = baseline_resting_hr + 14.0
        elif act == "moderate_walk":
            base_hr = baseline_resting_hr + 38.0
        else:
            base_hr = baseline_resting_hr + 50.0

        # Postprandial thermogenesis: slight HR elevation after meals
        recent_meals = np.sum(meals[max(0, t - 6):t + 1])
        meal_hr_boost = min(12.0, recent_meals * 0.08)

        hr_val = base_hr + meal_hr_boost + rng.normal(0, 2.2)
        heart_rates[t] = float(np.round(np.clip(hr_val, 48.0, 165.0), 1))

    df = pd.DataFrame({
        "timestamp": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in timestamps],
        "patient_id": patient_id,
        "glucose": glucose,
        "heart_rate": heart_rates,
        "steps": steps,
        "sleep_hours": np.round(sleep_hours_series, 2),
        "sleep_quality": np.round(sleep_quality_series, 1),
        "carbs_intake": np.round(meals, 1),
        "activity_type": activity_types,
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wearable sensor stream for T2D patient.")
    parser.add_argument("--patient_id", type=str, default="PT-SYNTH-001", help="Target patient ID")
    parser.add_argument("--ehr_path", type=str, default="data/raw/synthetic_ehr_patients.csv", help="Path to EHR CSV")
    parser.add_argument("--days", type=int, default=60, help="Days of continuous time-series")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_wearable_stream.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load or generate patient EHR profile
    if os.path.exists(args.ehr_path):
        df_ehr = pd.read_csv(args.ehr_path)
        matching = df_ehr[df_ehr["patient_id"] == args.patient_id]
        if not matching.empty:
            patient_row = matching.iloc[0].to_dict()
        else:
            patient_row = df_ehr.iloc[0].to_dict()
    else:
        patient_row = {
            "patient_id": args.patient_id,
            "name": "Arthur Pendelton",
            "age": 58,
            "gender": "Male",
            "bmi": 29.8,
            "hba1c": 8.1,
            "fasting_glucose": 138,
            "systolic_bp": 134,
            "diastolic_bp": 86,
        }

    df_stream = generate_patient_timeseries(patient_row, days=args.days, seed=args.seed)
    df_stream.to_csv(args.output, index=False)
    print(f"Generated {len(df_stream)} wearable sensor readings ({args.days} days) for {patient_row.get('name', 'Patient')} -> {args.output}")
