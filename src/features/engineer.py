"""
Feature Engineering Engine for Short-Term Glucose Spike Prediction.
Enforces strict chronological causality: every feature is computed from historical
observations [t - window, t] to prevent temporal data leakage.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Canonical list of model features
FEATURE_COLUMNS = [
    # Glucose kinematics
    "glucose_current",
    "glucose_delta_15m",
    "glucose_delta_30m",
    "glucose_delta_60m",
    "glucose_acceleration",
    "glucose_mean_1h",
    "glucose_std_1h",
    "glucose_mean_2h",
    "glucose_std_2h",
    "glucose_cv_2h",
    # Autonomic & cardiovascular
    "hr_current",
    "hr_mean_30m",
    "hr_delta_30m",
    # Activity & exertion
    "steps_current_15m",
    "steps_last_30m",
    "steps_last_60m",
    # Sleep
    "sleep_hours",
    "sleep_quality",
    # Nutrition / intake
    "carbs_current",
    "carbs_last_2h",
    # Circadian & temporal
    "time_of_day_sin",
    "time_of_day_cos",
    # Clinical EHR phenotype
    "age",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "systolic_bp",
    "medication_metformin",
    "medication_insulin",
    "diabetes_duration_years",
]

TARGET_COLUMN = "glucose_spike_2h"


def compute_spike_target(
    glucose_series: pd.Series,
    horizon_ticks: int = 8,  # 8 * 15 min = 120 min (2 hours)
    hyperglycemic_threshold: float = 180.0,
    surge_threshold: float = 50.0,
) -> pd.Series:
    """
    Computes binary ground-truth target: 1 if glucose in (t, t + 2h] >= 180 or rises by >= 50 mg/dL.
    """
    future_max = glucose_series.iloc[::-1].rolling(window=horizon_ticks, min_periods=1).max().iloc[::-1].shift(-1)
    spike_absolute = future_max >= hyperglycemic_threshold
    spike_relative = (future_max - glucose_series) >= surge_threshold
    target = (spike_absolute | spike_relative).astype(int)
    return target


def build_feature_matrix(
    df_stream: pd.DataFrame,
    patient_profile: Optional[Dict] = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Transforms continuous time-series stream and static EHR into ML feature vectors.
    Strictly retrospective: no future values are accessed during feature construction.
    """
    df = df_stream.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Glucose Kinematics (strictly backward-looking shift/rolling)
    g = df["glucose"]
    df["glucose_current"] = g
    df["glucose_delta_15m"] = g - g.shift(1)
    df["glucose_delta_30m"] = g - g.shift(2)
    df["glucose_delta_60m"] = g - g.shift(4)
    # Glucose acceleration: change in velocity
    v1 = g - g.shift(1)
    v2 = g.shift(1) - g.shift(2)
    df["glucose_acceleration"] = v1 - v2

    df["glucose_mean_1h"] = g.rolling(window=4, min_periods=1).mean()
    df["glucose_std_1h"] = g.rolling(window=4, min_periods=1).std().fillna(0.0)
    df["glucose_mean_2h"] = g.rolling(window=8, min_periods=1).mean()
    df["glucose_std_2h"] = g.rolling(window=8, min_periods=1).std().fillna(0.0)
    df["glucose_cv_2h"] = (df["glucose_std_2h"] / (df["glucose_mean_2h"] + 1e-5)) * 100.0

    # 2. Cardiovascular & Autonomic
    hr = df["heart_rate"]
    df["hr_current"] = hr
    df["hr_mean_30m"] = hr.rolling(window=2, min_periods=1).mean()
    df["hr_delta_30m"] = hr - hr.shift(2)

    # 3. Physical Activity
    steps = df["steps"]
    df["steps_current_15m"] = steps
    df["steps_last_30m"] = steps.rolling(window=2, min_periods=1).sum()
    df["steps_last_60m"] = steps.rolling(window=4, min_periods=1).sum()

    # 4. Nutrition
    carbs = df.get("carbs_intake", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["carbs_current"] = carbs
    df["carbs_last_2h"] = carbs.rolling(window=8, min_periods=1).sum()

    # 5. Circadian Signals
    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["time_of_day_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # 6. Static EHR Features (broadcast across time steps)
    ehr = patient_profile or {}
    df["age"] = float(ehr.get("age", 58))
    df["bmi"] = float(ehr.get("bmi", 29.5))
    df["hba1c"] = float(ehr.get("hba1c", 7.8))
    df["fasting_glucose"] = float(ehr.get("fasting_glucose", 135))
    df["systolic_bp"] = float(ehr.get("systolic_bp", 132))
    df["medication_metformin"] = int(ehr.get("medication_metformin", 1))
    df["medication_insulin"] = int(ehr.get("medication_insulin", 0))
    df["diabetes_duration_years"] = float(ehr.get("diabetes_duration_years", 5.0))

    if include_target:
        df[TARGET_COLUMN] = compute_spike_target(df["glucose"], horizon_ticks=8)
        # Drop the last 8 rows where the future 2-hour window is incomplete
        df = df.iloc[:-8].copy()

    # Drop warm-up rows (first 8 rows) to ensure stable rolling features
    df = df.iloc[8:].copy()

    # Fill any remaining NaNs
    df = df.fillna(0.0)
    return df


def extract_single_step_features(
    recent_stream_buffer: pd.DataFrame,
    patient_profile: Dict,
) -> pd.DataFrame:
    """
    Extracts the feature vector for the *latest* single tick from an active buffer.
    Used in real-time Digital Twin state inference.
    """
    df_features = build_feature_matrix(
        df_stream=recent_stream_buffer,
        patient_profile=patient_profile,
        include_target=False,
    )
    latest_row = df_features[FEATURE_COLUMNS].iloc[[-1]]
    return latest_row


def create_temporal_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into train, validation, and test subsets.
    Crucial: Shuffling is prohibited in time-series to prevent future leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy().reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    df_test = df.iloc[val_end:].copy().reset_index(drop=True)

    return df_train, df_val, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature matrix and temporal splits.")
    parser.add_argument("--stream_path", type=str, default="data/raw/synthetic_wearable_stream.csv")
    parser.add_argument("--ehr_path", type=str, default="data/raw/synthetic_ehr_patients.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load EHR and Stream
    df_ehr = pd.read_csv(args.ehr_path) if os.path.exists(args.ehr_path) else None
    patient_dict = df_ehr.iloc[0].to_dict() if df_ehr is not None else {}

    df_raw = pd.read_csv(args.stream_path)
    df_features = build_feature_matrix(df_raw, patient_profile=patient_dict, include_target=True)

    df_train, df_val, df_test = create_temporal_splits(df_features)

    df_train.to_csv(os.path.join(args.output_dir, "train_features.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, "val_features.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, "test_features.csv"), index=False)

    # Save feature columns metadata
    with open(os.path.join(args.output_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"Features created successfully:")
    print(f"  Train: {len(df_train)} rows | Positives: {df_train[TARGET_COLUMN].sum()} ({df_train[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"  Val:   {len(df_val)} rows | Positives: {df_val[TARGET_COLUMN].sum()} ({df_val[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"  Test:  {len(df_test)} rows | Positives: {df_test[TARGET_COLUMN].sum()} ({df_test[TARGET_COLUMN].mean()*100:.1f}%)")
