"""
Model evaluation module for Type 2 Diabetes Spike Prediction.
Computes ROC-AUC, PR-AUC, Confusion Matrix, Sensitivity/Recall, Specificity, F1,
and provides clinically meaningful interpretation of False Positives and False Negatives.
"""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates binary classification model performance across standard and healthcare metrics.

    Parameters:
        y_true: True binary target array (0 or 1).
        y_pred_proba: Predicted probability of positive class (glucose spike).
        threshold: Classification cutoff.

    Returns:
        Dictionary containing metrics, confusion matrix, and clinical diagnostics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Standard metrics
    roc_auc = float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Clinical trade-off analysis
    clinical_notes = {
        "false_positive_analysis": (
            f"There were {int(fp)} false positives. In clinical diabetes monitoring, "
            "a false positive triggers a cautionary notification when glucose does not spike. "
            "While it may induce minor alarm fatigue, it rarely poses acute physical risk."
        ),
        "false_negative_analysis": (
            f"There were {int(fn)} false negatives. In clinical practice, missing an acute "
            "hyperglycemic excursion (>= 180 mg/dL) is dangerous as it prevents timely behavioral "
            "or physical mitigation. Higher recall/sensitivity is clinically prioritized."
        ),
        "total_test_samples": int(len(y_true)),
        "actual_spike_prevalence": float(np.mean(y_true)),
    }

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "threshold_used": threshold,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "clinical_diagnostics": clinical_notes,
    }
