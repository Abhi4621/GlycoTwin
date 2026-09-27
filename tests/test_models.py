"""
Tests for Model Evaluation and SHAP Explainability Engine.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.engineer import FEATURE_COLUMNS
from src.models.evaluate import evaluate_binary_classifier
from src.models.explain import ModelExplainer


def test_evaluate_binary_classifier():
    """Tests evaluation metrics calculation and error analysis."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.85])

    metrics = evaluate_binary_classifier(y_true, y_probs, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
    assert "clinical_diagnostics" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_model_explainer():
    """Tests that ModelExplainer decomposes features into positive/negative drivers."""
    # Mock data and model
    X_mock = pd.DataFrame(np.random.randn(50, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y_mock = np.random.randint(0, 2, size=50)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_mock, y_mock)

    explainer = ModelExplainer(model=rf, feature_names=FEATURE_COLUMNS, background_data=X_mock)
    single_sample = X_mock.iloc[[0]]

    exps = explainer.explain_instance(single_sample, top_k=4)
    assert len(exps) == 4
    for exp in exps:
        assert "feature" in exp
        assert "label" in exp
        assert "direction" in exp
        assert exp["direction"] in ["increases_risk", "decreases_risk"]
