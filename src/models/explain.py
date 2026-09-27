"""
Model Explainability & Risk Attribution Engine.
Uses SHAP (SHapley Additive exPlanations) and tree-attribution mechanisms
to decompose individual spike predictions into clinically intuitive driving factors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# Clinician-friendly mappings for feature explanations
FEATURE_CLINICAL_LABELS = {
    "glucose_current": "Elevated current glucose level",
    "glucose_delta_15m": "Fast 15-min glucose upward velocity",
    "glucose_delta_30m": "Steep 30-min glucose rise",
    "glucose_delta_60m": "Sustained 60-min glucose upward trajectory",
    "glucose_acceleration": "Accelerating glucose trajectory",
    "glucose_mean_1h": "High 1-hour average glucose",
    "glucose_std_1h": "Elevated short-term glycemic instability",
    "glucose_mean_2h": "High 2-hour baseline glucose",
    "glucose_std_2h": "High 2-hour glycemic variability",
    "glucose_cv_2h": "Elevated coefficient of variation (%CV)",
    "hr_current": "Elevated instantaneous heart rate",
    "hr_mean_30m": "Sustained autonomic heart rate elevation",
    "hr_delta_30m": "Recent sympathetic heart rate surge",
    "steps_current_15m": "Recent physical activity / steps",
    "steps_last_30m": "Post-meal physical movement (last 30 min)",
    "steps_last_60m": "Total activity volume over past hour",
    "sleep_hours": "Prior night sleep duration",
    "sleep_quality": "Prior night sleep quality score",
    "carbs_current": "Recent carbohydrate meal ingestion",
    "carbs_last_2h": "Cumulative carbohydrate intake (last 2h)",
    "time_of_day_sin": "Circadian time effect",
    "time_of_day_cos": "Circadian rhythm phase",
    "age": "Patient age factor",
    "bmi": "Patient BMI / adiposity resistance",
    "hba1c": "Baseline clinical HbA1c",
    "fasting_glucose": "Baseline fasting glucose profile",
    "systolic_bp": "Comorbid cardiovascular systolic BP",
    "medication_metformin": "Metformin therapy adherence",
    "medication_insulin": "Basal insulin treatment regimen",
    "diabetes_duration_years": "Chronicity of Type 2 Diabetes",
}


class ModelExplainer:
    """
    Computes local feature attributions for individual real-time patient predictions.
    Translates mathematical SHAP values into clinically meaningful doctor explanations.
    """

    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.shap_explainer = None

        try:
            import shap
            if hasattr(model, "predict_proba"):
                # Prefer TreeExplainer for tree models
                if "Forest" in model.__class__.__name__ or "XGB" in model.__class__.__name__ or "Gradient" in model.__class__.__name__:
                    self.shap_explainer = shap.TreeExplainer(model)
                elif background_data is not None:
                    sample = background_data[feature_names].iloc[:50]
                    self.shap_explainer = shap.LinearExplainer(model, sample)
        except Exception:
            self.shap_explainer = None

    def explain_instance(
        self,
        features_df: pd.DataFrame,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Decomposes a single prediction into top positive and negative contributing factors.
        """
        row = features_df[self.feature_names].iloc[0]

        attributions = {}
        # Try computing SHAP values
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(features_df[self.feature_names])
                if isinstance(shap_values, list):
                    # For binary classification, take positive class (index 1)
                    vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif hasattr(shap_values, "values"):
                    vals = shap_values.values[0]
                else:
                    vals = shap_values[0]

                if vals.ndim > 1:
                    vals = vals[:, 1] if vals.shape[1] > 1 else vals[:, 0]

                for feat, val in zip(self.feature_names, vals):
                    attributions[feat] = float(val)
            except Exception:
                attributions = {}

        # Fallback: model weights or standardized feature deviations
        if not attributions:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                for feat, imp in zip(self.feature_names, importances):
                    # Direction based on value relative to typical thresholds
                    val = float(row[feat])
                    direction = 1.0 if ("delta" in feat or "glucose" in feat or "carbs" in feat) and val > 0 else -1.0
                    attributions[feat] = imp * direction
            elif hasattr(self.model, "coef_"):
                coefs = self.model.coef_[0]
                for feat, c in zip(self.feature_names, coefs):
                    attributions[feat] = float(c * row[feat])

        # Sort features by absolute attribution magnitude
        sorted_feats = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)

        explanations = []
        for feat_name, attr_val in sorted_feats[:top_k]:
            label = FEATURE_CLINICAL_LABELS.get(feat_name, feat_name.replace("_", " ").title())
            val = row[feat_name]
            direction = "increases_risk" if attr_val > 0 else "decreases_risk"
            impact_sign = "+" if attr_val > 0 else "-"

            # Formatting specific values for clinical clarity
            if "delta" in feat_name or "glucose" in feat_name:
                detail = f"{label} ({val:+.1f} mg/dL)"
            elif "sleep_hours" in feat_name:
                detail = f"{label} ({val:.1f} hrs)"
            elif "carbs" in feat_name:
                detail = f"{label} ({val:.0f}g)"
            elif "steps" in feat_name:
                detail = f"{label} ({int(val)} steps)"
            elif "hr" in feat_name:
                detail = f"{label} ({val:.0f} bpm)"
            else:
                detail = f"{label} ({val:.1f})"

            impact_pct = int(min(95, max(5, abs(attr_val) * 100)))

            explanations.append({
                "feature": feat_name,
                "label": detail,
                "contribution": f"{impact_sign}{impact_pct}%",
                "attribution_value": round(float(attr_val), 4),
                "direction": direction,
                "current_value": float(val),
            })

        return explanations
