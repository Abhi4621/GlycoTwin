"""
Digital Twin State Engine and Physiological Inference Coordinator.
Coordinates the live loop:
Sensor Ingestion -> Validation -> Digital Twin State Update -> Feature Generation ->
ML Spike Prediction -> Trajectory Projection -> SHAP Explanation -> Emitting State.
"""

import datetime
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features.engineer import FEATURE_COLUMNS, extract_single_step_features
from src.models.explain import ModelExplainer
from .ingestion import validate_sensor_reading
from .twin_state import PatientTwinState


class DigitalTwinEngine:
    """
    Core engine managing the in-silico Digital Twin for a Type 2 Diabetes patient.
    """

    def __init__(
        self,
        patient_id: str,
        ehr_profile: Dict[str, Any],
        model_path: Optional[str] = "models/deployed_model_package.joblib",
    ):
        self.patient_id = patient_id
        self.ehr_profile = ehr_profile
        self.state = PatientTwinState(patient_id=patient_id, ehr_profile=ehr_profile)
        self.model = None
        self.scaler = None
        self.uses_scaler = False
        self.explainer: Optional[ModelExplainer] = None

        self._load_or_init_model(model_path)

    def _load_or_init_model(self, model_path: Optional[str]) -> None:
        """Loads trained ML package or initializes an intelligent clinical heuristic model."""
        if model_path and os.path.exists(model_path):
            try:
                pkg = joblib.load(model_path)
                if isinstance(pkg, dict):
                    self.model = pkg.get("model")
                    self.uses_scaler = pkg.get("uses_scaler", False)
                else:
                    self.model = pkg
            except Exception:
                self.model = None

        # Try fallback to best_model_xgboost or baseline_random_forest
        if self.model is None:
            for alt in ["models/best_model_xgboost.joblib", "models/baseline_random_forest.joblib"]:
                if os.path.exists(alt):
                    try:
                        self.model = joblib.load(alt)
                        break
                    except Exception:
                        pass

        # If models directory does not exist or has not been trained yet,
        # initialize a trained lightweight surrogate model on physiological rules
        if self.model is None:
            self._init_surrogate_model()

        # Load scaler if applicable
        scaler_path = "models/scaler.joblib"
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                self.scaler = None

        # Initialize explainability engine
        self.explainer = ModelExplainer(model=self.model, feature_names=FEATURE_COLUMNS)

    def _init_surrogate_model(self) -> None:
        """Trains an instant surrogate tree model to ensure zero cold-start failures."""
        rng = np.random.default_rng(42)
        n_synth = 400
        # Synthetic design matrix for warm-up
        X_mock = np.zeros((n_synth, len(FEATURE_COLUMNS)))
        # Target: probability rises with glucose delta and current glucose
        g_curr = rng.uniform(80, 240, n_synth)
        d_30 = rng.normal(0, 20, n_synth)
        carbs = rng.uniform(0, 80, n_synth)
        sleep = rng.uniform(4.5, 9.0, n_synth)

        logits = 0.02 * (g_curr - 130) + 0.06 * d_30 + 0.03 * carbs - 0.25 * (sleep - 7.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_mock = (probs > 0.45).astype(int)

        X_mock[:, 0] = g_curr
        X_mock[:, 2] = d_30
        X_mock[:, 18] = carbs
        X_mock[:, 16] = sleep

        clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        clf.fit(X_mock, y_mock)
        self.model = clf

    def seed_initial_history(self, historical_df: pd.DataFrame) -> None:
        """Seeds the twin state with historical data so the dashboard has rich baseline history."""
        # Take the most recent 48 to 96 ticks
        records = historical_df.tail(64).to_dict("records")
        for rec in records:
            validated, _ = validate_sensor_reading(rec)
            self.state.add_observation(validated)

        # Run an initial evaluation on latest state
        if len(self.state.telemetry_buffer) >= 8:
            self._evaluate_current_state()

    def process_observation(self, raw_reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full Digital Twin cycle for an incoming live sensor tick:
        Validate -> Buffer Update -> Feature Extraction -> ML Inference ->
        Trajectory Projection -> Explainability -> Return State.
        """
        # 1. Validation
        sanitized_obs, warnings = validate_sensor_reading(raw_reading)

        # 2. Add to virtual patient buffer
        self.state.add_observation(sanitized_obs)

        # 3. Predict & Project
        eval_result = self._evaluate_current_state()

        # 4. Return serialized state with any validation warnings
        response = self.state.to_dict()
        response["validation_warnings"] = warnings
        return response

    def _evaluate_current_state(self) -> Dict[str, Any]:
        """Runs feature engineering, model inference, trajectory projection, and SHAP explainability."""
        buf_df = self.state.get_buffer_dataframe()
        if len(buf_df) < 4:
            # Need a minimal buffer length for velocity; return conservative baseline
            return {}

        # 1. Extract feature row
        features_df = extract_single_step_features(buf_df, self.ehr_profile)

        # 2. Run model inference
        X_input = features_df[FEATURE_COLUMNS]
        if self.uses_scaler and self.scaler is not None:
            X_input_scaled = self.scaler.transform(X_input)
            spike_prob = float(self.model.predict_proba(X_input_scaled)[0, 1])
        else:
            spike_prob = float(self.model.predict_proba(X_input)[0, 1])

        # 3. Stratify Risk
        if spike_prob < 0.35:
            risk_level = "Low"
        elif spike_prob < 0.65:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # 4. Project 2-hour trajectory
        latest_g = float(self.state.latest_observation["glucose"])
        vel_per_min = self.state.derived_state["glucose_velocity_mgdl_per_min"]
        last_ts = pd.to_datetime(self.state.latest_observation["timestamp"])

        pred_traj: List[float] = []
        conf_low: List[float] = []
        conf_high: List[float] = []
        traj_times: List[str] = []

        curr_sim_g = latest_g
        # Extrapolate over 8 ticks (15 min each = 120 min)
        for step in range(1, 9):
            step_time = last_ts + datetime.timedelta(minutes=15 * step)
            traj_times.append(step_time.strftime("%H:%M"))

            # Momentum decays with time, pulling towards homeostatic or spike target
            decay = np.exp(-0.15 * step)
            # If high spike probability, projected path accelerates upwards
            risk_boost = (spike_prob - 0.30) * 18.0 * (1.0 - np.exp(-0.25 * step))
            step_vel = vel_per_min * 15.0 * decay + risk_boost

            curr_sim_g = float(np.clip(curr_sim_g + step_vel, 60.0, 360.0))
            pred_traj.append(curr_sim_g)

            # Widening confidence band over 2-hour horizon
            sigma = 4.0 + 3.0 * step
            conf_low.append(max(50.0, curr_sim_g - sigma))
            conf_high.append(min(380.0, curr_sim_g + sigma))

        # 5. Explainability (SHAP / Feature Attribution)
        risk_factors = []
        if self.explainer is not None:
            risk_factors = self.explainer.explain_instance(features_df, top_k=4)

        # 6. Update Twin state
        self.state.update_prediction(
            spike_prob=spike_prob,
            risk_level=risk_level,
            predicted_trajectory=pred_traj,
            trajectory_timestamps=traj_times,
            confidence_lower=conf_low,
            confidence_upper=conf_high,
            risk_factors=risk_factors,
        )

        return self.state.latest_prediction
