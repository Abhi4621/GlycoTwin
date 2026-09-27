"""
Patient Digital Twin State Representation.
Maintains the continuous in-silico physiological twin including EHR baseline,
rolling observation buffer, derived glycemic dynamics, and latest risk prediction.
"""

from collections import deque
import datetime
from typing import Any, Deque, Dict, List, Optional
import numpy as np
import pandas as pd


class PatientTwinState:
    """
    State container for a single virtual patient's Digital Twin.
    """

    def __init__(
        self,
        patient_id: str,
        ehr_profile: Dict[str, Any],
        buffer_maxlen: int = 96,  # 96 * 15m = 24 hours of temporal memory
    ):
        self.patient_id = patient_id
        self.ehr_profile = ehr_profile
        self.buffer_maxlen = buffer_maxlen
        self.telemetry_buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_maxlen)
        self.latest_observation: Optional[Dict[str, Any]] = None
        self.derived_state: Dict[str, Any] = {
            "glucose_velocity_mgdl_per_min": 0.0,
            "glucose_delta_30m": 0.0,
            "glucose_mean_2h": 120.0,
            "glucose_std_2h": 0.0,
            "cv_percent": 0.0,
            "autonomic_hr_elevation": 0.0,
            "active_steps_2h": 0,
        }
        self.latest_prediction: Dict[str, Any] = {
            "spike_probability_2h": 0.15,
            "risk_level": "Low",
            "predicted_trajectory": [],
            "trajectory_timestamps": [],
            "confidence_lower": [],
            "confidence_upper": [],
            "risk_factors": [],
            "timestamp": None,
        }
        # Timeline recording snapshots of twin evolution
        self.state_history_snapshots: List[Dict[str, Any]] = []

    def add_observation(self, observation: Dict[str, Any]) -> None:
        """Appends a validated sensor observation to the rolling buffer."""
        self.telemetry_buffer.append(observation)
        self.latest_observation = observation
        self._update_derived_state()

    def _update_derived_state(self) -> None:
        """Recalculates rolling physiological biomarkers based on buffer contents."""
        if not self.telemetry_buffer:
            return

        buf = list(self.telemetry_buffer)
        glucoses = [obs["glucose"] for obs in buf]
        hrs = [obs["heart_rate"] for obs in buf]
        steps = [obs["steps"] for obs in buf]

        # Instantaneous velocity (mg/dL per minute)
        if len(glucoses) >= 2:
            velocity_15m = (glucoses[-1] - glucoses[-2]) / 15.0
        else:
            velocity_15m = 0.0

        # 30-min glucose change
        if len(glucoses) >= 3:
            delta_30m = glucoses[-1] - glucoses[-3]
        elif len(glucoses) >= 2:
            delta_30m = glucoses[-1] - glucoses[-2]
        else:
            delta_30m = 0.0

        # 2-hour rolling statistics (last 8 readings)
        window_g = glucoses[-8:] if len(glucoses) >= 8 else glucoses
        mean_2h = float(np.mean(window_g))
        std_2h = float(np.std(window_g))
        cv_pct = float((std_2h / (mean_2h + 1e-5)) * 100.0)

        # Cardiovascular & physical activity metrics
        window_steps = steps[-8:] if len(steps) >= 8 else steps
        sum_steps_2h = int(np.sum(window_steps))

        baseline_hr = 68.0
        current_hr = hrs[-1] if hrs else baseline_hr
        hr_elev = float(max(0.0, current_hr - baseline_hr))

        self.derived_state = {
            "glucose_velocity_mgdl_per_min": round(velocity_15m, 2),
            "glucose_delta_30m": round(delta_30m, 1),
            "glucose_mean_2h": round(mean_2h, 1),
            "glucose_std_2h": round(std_2h, 1),
            "cv_percent": round(cv_pct, 1),
            "autonomic_hr_elevation": round(hr_elev, 1),
            "active_steps_2h": sum_steps_2h,
        }

    def update_prediction(
        self,
        spike_prob: float,
        risk_level: str,
        predicted_trajectory: List[float],
        trajectory_timestamps: List[str],
        confidence_lower: List[float],
        confidence_upper: List[float],
        risk_factors: List[Dict[str, Any]],
    ) -> None:
        """Stores the newly computed prediction and risk factors."""
        ts = self.latest_observation.get("timestamp") if self.latest_observation else datetime.datetime.now().isoformat()
        self.latest_prediction = {
            "spike_probability_2h": round(spike_prob, 3),
            "risk_level": risk_level,
            "predicted_trajectory": [round(x, 1) for x in predicted_trajectory],
            "trajectory_timestamps": trajectory_timestamps,
            "confidence_lower": [round(x, 1) for x in confidence_lower],
            "confidence_upper": [round(x, 1) for x in confidence_upper],
            "risk_factors": risk_factors,
            "timestamp": ts,
        }

        # Snapshot for day progression
        self.state_history_snapshots.append({
            "timestamp": ts,
            "glucose": self.latest_observation.get("glucose") if self.latest_observation else None,
            "heart_rate": self.latest_observation.get("heart_rate") if self.latest_observation else None,
            "steps": self.latest_observation.get("steps") if self.latest_observation else 0,
            "spike_probability_2h": round(spike_prob, 3),
            "risk_level": risk_level,
            "glucose_velocity": self.derived_state.get("glucose_velocity_mgdl_per_min", 0.0),
        })
        if len(self.state_history_snapshots) > 96:
            self.state_history_snapshots.pop(0)

    def get_buffer_dataframe(self) -> pd.DataFrame:
        """Returns the current rolling observation buffer as a pandas DataFrame."""
        if not self.telemetry_buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self.telemetry_buffer))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current twin state for API responses and dashboard consumption."""
        return {
            "patient_id": self.patient_id,
            "ehr_profile": self.ehr_profile,
            "latest_observation": self.latest_observation,
            "derived_state": self.derived_state,
            "prediction": self.latest_prediction,
            "buffer_length": len(self.telemetry_buffer),
            "recent_history_preview": list(self.telemetry_buffer)[-12:] if self.telemetry_buffer else [],
        }
