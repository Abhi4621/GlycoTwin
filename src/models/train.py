"""
Model Training and Comparison Pipeline for Short-Term Glucose Spike Prediction.
Trains Logistic Regression, Random Forest, and Gradient Boosted Trees (XGBoost),
compares validation performance, and saves benchmark artifacts.
"""

import argparse
import json
import os
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.engineer import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.evaluate import evaluate_binary_classifier


def train_and_compare_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = "models",
) -> Dict[str, Any]:
    """
    Trains multiple candidate models, evaluates on validation and test sets,
    and persists trained artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN].values

    X_val = df_val[FEATURE_COLUMNS]
    y_val = df_val[TARGET_COLUMN].values

    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[TARGET_COLUMN].values

    # Fit feature scaler for models sensitive to scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    # Compute class weight for imbalance
    pos_weight = float((len(y_train) - sum(y_train)) / max(1, sum(y_train)))

    # 1. Model 1: Logistic Regression (Interpretable linear baseline)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    val_probs_lr = lr.predict_proba(X_val_scaled)[:, 1]
    test_probs_lr = lr.predict_proba(X_test_scaled)[:, 1]

    # 2. Model 2: Random Forest (Nonlinear ensemble baseline)
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    val_probs_rf = rf.predict_proba(X_val)[:, 1]
    test_probs_rf = rf.predict_proba(X_test)[:, 1]

    # 3. Model 3: Gradient Boosted Trees (XGBoost or Sklearn GradientBoosting)
    try:
        import xgboost as xgb
        gb = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            scale_pos_weight=pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss",
        )
        gb.fit(X_train, y_train)
        val_probs_gb = gb.predict_proba(X_val)[:, 1]
        test_probs_gb = gb.predict_proba(X_test)[:, 1]
        model_name_gb = "XGBoost"
    except ImportError:
        gb = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
        gb.fit(X_train, y_train)
        val_probs_gb = gb.predict_proba(X_val)[:, 1]
        test_probs_gb = gb.predict_proba(X_test)[:, 1]
        model_name_gb = "GradientBoosting (Sklearn)"

    # Evaluate all models on validation set
    val_metrics = {
        "LogisticRegression": evaluate_binary_classifier(y_val, val_probs_lr),
        "RandomForest": evaluate_binary_classifier(y_val, val_probs_rf),
        model_name_gb: evaluate_binary_classifier(y_val, val_probs_gb),
    }

    # Evaluate all models on out-of-time test set
    test_metrics = {
        "LogisticRegression": evaluate_binary_classifier(y_test, test_probs_lr),
        "RandomForest": evaluate_binary_classifier(y_test, test_probs_rf),
        model_name_gb: evaluate_binary_classifier(y_test, test_probs_gb),
    }

    # Determine best model based on validation PR-AUC and ROC-AUC
    candidates = [
        ("LogisticRegression", val_metrics["LogisticRegression"]["pr_auc"], lr, True),
        ("RandomForest", val_metrics["RandomForest"]["pr_auc"], rf, False),
        (model_name_gb, val_metrics[model_name_gb]["pr_auc"], gb, False),
    ]
    best_name, best_prauc, best_model, uses_scaler = max(candidates, key=lambda x: x[1])

    # Save models
    joblib.dump(lr, os.path.join(output_dir, "baseline_logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(output_dir, "baseline_random_forest.joblib"))
    joblib.dump(gb, os.path.join(output_dir, "best_model_xgboost.joblib"))

    # Also save the designated primary deployment model
    deployment_package = {
        "model": best_model,
        "model_name": best_name,
        "uses_scaler": uses_scaler,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(deployment_package, os.path.join(output_dir, "deployed_model_package.joblib"))

    # Save feature metadata
    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # Save comparison metrics summary
    summary = {
        "best_model_selected": best_name,
        "validation_comparison": val_metrics,
        "test_benchmark_results": test_metrics,
        "dataset_split_summary": {
            "train_samples": int(len(df_train)),
            "val_samples": int(len(df_val)),
            "test_samples": int(len(df_test)),
            "spike_prevalence_train": float(df_train[TARGET_COLUMN].mean()),
            "spike_prevalence_test": float(df_test[TARGET_COLUMN].mean()),
        },
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark glucose spike prediction models.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()

    train_path = os.path.join(args.processed_dir, "train_features.csv")
    val_path = os.path.join(args.processed_dir, "val_features.csv")
    test_path = os.path.join(args.processed_dir, "test_features.csv")

    # If processed features don't exist yet, build them
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Processed splits not found. Auto-generating baseline data and features...")
        from src.data.loader import load_patient_data
        from src.features.engineer import build_feature_matrix, create_temporal_splits

        patient_dict, df_stream = load_patient_data()
        df_feat = build_feature_matrix(df_stream, patient_profile=patient_dict, include_target=True)
        df_train, df_val, df_test = create_temporal_splits(df_feat)
        os.makedirs(args.processed_dir, exist_ok=True)
        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)
    else:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

    summary = train_and_compare_models(df_train, df_val, df_test, output_dir=args.output_dir)
    print("=" * 60)
    print(f"Model Training Complete! Selected Best Model: {summary['best_model_selected']}")
    print("=" * 60)
    for model_name, metrics in summary["test_benchmark_results"].items():
        print(f"[{model_name}] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Sensitivity: {metrics['recall_sensitivity']:.4f} | F1: {metrics['f1_score']:.4f}")
