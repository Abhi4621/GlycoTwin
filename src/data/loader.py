"""
Data loader and caching utilities for Type 2 Diabetes Digital Twin.
Gracefully auto-generates baseline cohorts if raw files are missing.
"""

import os
from typing import Dict, Optional, Tuple
import pandas as pd

from .generate_ehr import generate_ehr_cohort
from .generate_timeseries import generate_patient_timeseries


DEFAULT_EHR_PATH = "data/raw/synthetic_ehr_patients.csv"
DEFAULT_STREAM_PATH = "data/raw/synthetic_wearable_stream.csv"
DEFAULT_PROCESSED_DIR = "data/processed"


def ensure_raw_datasets(
    ehr_path: str = DEFAULT_EHR_PATH,
    stream_path: str = DEFAULT_STREAM_PATH,
    num_patients: int = 100,
    days: int = 60,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensures raw EHR and wearable stream files exist on disk, generating them if needed."""
    os.makedirs(os.path.dirname(ehr_path), exist_ok=True)
    os.makedirs(os.path.dirname(stream_path), exist_ok=True)

    if os.path.exists(ehr_path):
        df_ehr = pd.read_csv(ehr_path)
    else:
        df_ehr = generate_ehr_cohort(num_patients=num_patients, seed=seed)
        df_ehr.to_csv(ehr_path, index=False)

    if os.path.exists(stream_path):
        df_stream = pd.read_csv(stream_path)
    else:
        patient_row = df_ehr.iloc[0].to_dict()
        df_stream = generate_patient_timeseries(patient_row, days=days, seed=seed)
        df_stream.to_csv(stream_path, index=False)

    return df_ehr, df_stream


def load_patient_data(
    patient_id: str = "PT-SYNTH-001",
    ehr_path: str = DEFAULT_EHR_PATH,
    stream_path: str = DEFAULT_STREAM_PATH,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Loads EHR profile and continuous wearable stream for a specific patient.
    """
    df_ehr, df_stream = ensure_raw_datasets(ehr_path=ehr_path, stream_path=stream_path)

    ehr_match = df_ehr[df_ehr["patient_id"] == patient_id]
    if not ehr_match.empty:
        patient_profile = ehr_match.iloc[0].to_dict()
    else:
        patient_profile = df_ehr.iloc[0].to_dict()

    patient_stream = df_stream[df_stream["patient_id"] == patient_id].copy()
    if patient_stream.empty:
        # Generate on the fly for this patient
        patient_stream = generate_patient_timeseries(patient_profile, days=45, seed=42)

    patient_stream["timestamp"] = pd.to_datetime(patient_stream["timestamp"])
    patient_stream = patient_stream.sort_values("timestamp").reset_index(drop=True)
    return patient_profile, patient_stream


def load_processed_splits(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads pre-split train, validation, and test feature DataFrames."""
    train_path = os.path.join(processed_dir, "train_features.csv")
    val_path = os.path.join(processed_dir, "val_features.csv")
    test_path = os.path.join(processed_dir, "test_features.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Processed feature splits not found in {processed_dir}. "
            f"Please run 'python -m src.features.engineer' to generate them."
        )

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_val, df_test
