"""
Sensor Data Ingestion and Clinical Boundary Validation Engine.
Guarantees integrity of continuous telemetry before ingestion into the Digital Twin.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple


class SensorValidationError(ValueError):
    """Raised when an incoming sensor reading violates physiological limits."""
    pass


# Clinically established physiological bounds
GLUCOSE_MIN = 20.0     # Extreme severe hypoglycemia limit
GLUCOSE_MAX = 600.0    # Extreme hyperosmolar hyperglycemic limit
HEART_RATE_MIN = 30.0  # Extreme bradycardia limit
HEART_RATE_MAX = 220.0 # Maximum physiological human heart rate
STEPS_MIN = 0
STEPS_MAX_15M = 8000   # Max human running cadence in 15 minutes


def validate_sensor_reading(raw_reading: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates and sanitizes a raw sensor observation dictionary.

    Parameters:
        raw_reading: Incoming raw dictionary containing sensor telemetry.

    Returns:
        Tuple of (sanitized_reading_dict, list_of_warning_messages)

    Raises:
        SensorValidationError if telemetry violates physical limits.
    """
    warnings: List[str] = []

    # 1. Timestamp validation
    ts_raw = raw_reading.get("timestamp")
    if not ts_raw:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        warnings.append("Missing timestamp; assigned current system time.")
    else:
        try:
            if isinstance(ts_raw, (datetime.datetime, datetime.date)):
                ts = ts_raw.strftime("%Y-%m-%d %H:%M:%S")
            else:
                parsed = datetime.datetime.fromisoformat(str(ts_raw).replace("Z", ""))
                ts = parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            warnings.append(f"Invalid timestamp format '{ts_raw}'; fallback to system time.")

    # 2. Glucose validation
    if "glucose" not in raw_reading or raw_reading["glucose"] is None:
        raise SensorValidationError("Incoming sensor payload missing required field: 'glucose'.")
    try:
        glucose = float(raw_reading["glucose"])
    except (ValueError, TypeError):
        raise SensorValidationError(f"Invalid non-numeric glucose value: {raw_reading.get('glucose')}")

    if glucose < GLUCOSE_MIN or glucose > GLUCOSE_MAX:
        raise SensorValidationError(
            f"Glucose reading {glucose} mg/dL is outside plausible human physiological limits "
            f"[{GLUCOSE_MIN}, {GLUCOSE_MAX}]."
        )
    if glucose < 54.0:
        warnings.append(f"Clinical Alert: Level 2 Hypoglycemia detected ({glucose} mg/dL).")
    elif glucose >= 250.0:
        warnings.append(f"Clinical Alert: Severe Hyperglycemia detected ({glucose} mg/dL).")

    # 3. Heart Rate validation
    hr_raw = raw_reading.get("heart_rate", 72.0)
    try:
        hr = float(hr_raw)
    except (ValueError, TypeError):
        hr = 72.0
        warnings.append("Non-numeric heart rate; imputed with resting 72 bpm.")

    if hr < HEART_RATE_MIN or hr > HEART_RATE_MAX:
        raise SensorValidationError(
            f"Heart rate {hr} bpm is outside plausible physiological limits [{HEART_RATE_MIN}, {HEART_RATE_MAX}]."
        )

    # 4. Activity steps validation
    steps_raw = raw_reading.get("steps", 0)
    try:
        steps = int(steps_raw)
    except (ValueError, TypeError):
        steps = 0
        warnings.append("Non-numeric step count; reset to 0.")

    if steps < STEPS_MIN or steps > STEPS_MAX_15M:
        warnings.append(f"Steps count {steps} outside 15-minute bounds [0, {STEPS_MAX_15M}]; clamped.")
        steps = int(min(max(steps, STEPS_MIN), STEPS_MAX_15M))

    # 5. Sleep attributes
    sleep_h = float(raw_reading.get("sleep_hours", 7.0))
    sleep_h = float(min(max(sleep_h, 0.0), 24.0))
    sleep_q = float(raw_reading.get("sleep_quality", 70.0))
    sleep_q = float(min(max(sleep_q, 0.0), 100.0))

    # 6. Carbs & activity type
    carbs = float(max(0.0, raw_reading.get("carbs_intake", 0.0)))
    act_type = str(raw_reading.get("activity_type", "sedentary"))

    sanitized = {
        "timestamp": ts,
        "patient_id": raw_reading.get("patient_id", "PT-SYNTH-001"),
        "glucose": round(glucose, 1),
        "heart_rate": round(hr, 1),
        "steps": steps,
        "sleep_hours": round(sleep_h, 2),
        "sleep_quality": round(sleep_q, 1),
        "carbs_intake": round(carbs, 1),
        "activity_type": act_type,
    }

    return sanitized, warnings
