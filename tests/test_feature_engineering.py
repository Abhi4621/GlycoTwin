"""
Tests for Temporal Feature Engineering and Data Leakage Prevention.
"""

import pandas as pd
import pytest

from src.features.engineer import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    compute_spike_target,
    create_temporal_splits,
)


def test_compute_spike_target():
    """Tests spike target logic over 2-hour forward horizon."""
    # Sequence of 10 ticks: flat 100, then at index 5 jumps to 190
    glucose = pd.Series([100, 102, 101, 105, 110, 190, 185, 160, 140, 120])
    target = compute_spike_target(glucose, horizon_ticks=3, hyperglycemic_threshold=180.0)

    # At index 2, 3, 4 the forward window looks ahead and hits 190 -> target should be 1
    assert target.iloc[2] == 1
    assert target.iloc[3] == 1
    assert target.iloc[4] == 1


def test_build_feature_matrix_and_splits():
    """Validates that build_feature_matrix generates all canonical features and chronological splits."""
    dates = pd.date_range("2026-06-01", periods=120, freq="15min")
    df_mock = pd.DataFrame({
        "timestamp": dates,
        "glucose": [110.0 + (i % 20) for i in range(120)],
        "heart_rate": [70.0 + (i % 10) for i in range(120)],
        "steps": [(i * 25) % 800 for i in range(120)],
        "sleep_hours": [7.0] * 120,
        "sleep_quality": [75.0] * 120,
        "carbs_intake": [0.0] * 120,
    })

    patient_dict = {"age": 55, "bmi": 28.0, "hba1c": 7.5, "fasting_glucose": 130}
    feat_df = build_feature_matrix(df_mock, patient_profile=patient_dict, include_target=True)

    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
    assert TARGET_COLUMN in feat_df.columns

    # Verify chronological splits
    df_train, df_val, df_test = create_temporal_splits(feat_df, train_ratio=0.7, val_ratio=0.15)
    assert len(df_train) + len(df_val) + len(df_test) == len(feat_df)
    # Check that train max timestamp precedes val min timestamp (No leakage!)
    assert df_train["timestamp"].max() < df_val["timestamp"].min()
    assert df_val["timestamp"].max() < df_test["timestamp"].min()
