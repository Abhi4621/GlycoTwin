"""
Feature engineering module for Type 2 Diabetes Digital Twin.
Transforms temporal sensor streams and EHR phenotypes into leak-free feature vectors.
"""

from .engineer import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    create_temporal_splits,
    extract_single_step_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "build_feature_matrix",
    "create_temporal_splits",
    "extract_single_step_features",
]
