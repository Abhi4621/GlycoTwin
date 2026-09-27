"""
Clinical Simulation Scenarios for Doctor Dashboard Demonstration.
Encapsulates realistic physiological progressions showing how the Digital Twin responds.
"""

from typing import Any, Dict, List


CLINICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "postprandial_spike": {
        "title": "Postprandial Glucose Surge (Sedentary after 75g Carbs)",
        "description": (
            "Arthur finishes a heavy carbohydrate lunch without subsequent movement. "
            "Poor prior sleep (5.1 hrs) elevates insulin resistance. "
            "Watch glucose climb rapidly and trigger High Predicted Spike Risk (> 80%)."
        ),
        "expected_risk": "High (> 75%)",
        "ticks": [
            {"time_offset": "10:00", "glucose": 118.0, "heart_rate": 72.0, "steps": 45, "carbs": 75.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "10:15", "glucose": 126.0, "heart_rate": 76.0, "steps": 30, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "10:30", "glucose": 139.0, "heart_rate": 79.0, "steps": 25, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "10:45", "glucose": 151.0, "heart_rate": 83.0, "steps": 40, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "11:00", "glucose": 168.0, "heart_rate": 86.0, "steps": 15, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "11:15", "glucose": 184.0, "heart_rate": 88.0, "steps": 20, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "11:30", "glucose": 198.0, "heart_rate": 85.0, "steps": 10, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
            {"time_offset": "11:45", "glucose": 206.0, "heart_rate": 81.0, "steps": 35, "carbs": 0.0, "sleep_hours": 5.1, "sleep_quality": 58.0, "activity": "sedentary"},
        ],
    },
    "euglycemic_walk": {
        "title": "Stable Euglycemic Control (Post-Meal Light Walk)",
        "description": (
            "Arthur consumes a balanced 45g lunch and immediately takes an active 20-minute walk. "
            "Contraction-mediated GLUT4 translocation clears glucose into skeletal muscle, "
            "blunting excursion and keeping risk Low (< 25%)."
        ),
        "expected_risk": "Low (< 25%)",
        "ticks": [
            {"time_offset": "12:30", "glucose": 114.0, "heart_rate": 74.0, "steps": 150, "carbs": 45.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "light_active"},
            {"time_offset": "12:45", "glucose": 122.0, "heart_rate": 108.0, "steps": 1420, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "moderate_walk"},
            {"time_offset": "13:00", "glucose": 126.0, "heart_rate": 104.0, "steps": 1180, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "moderate_walk"},
            {"time_offset": "13:15", "glucose": 123.0, "heart_rate": 82.0, "steps": 310, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "light_active"},
            {"time_offset": "13:30", "glucose": 118.0, "heart_rate": 75.0, "steps": 90, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "sedentary"},
            {"time_offset": "13:45", "glucose": 115.0, "heart_rate": 71.0, "steps": 60, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "sedentary"},
            {"time_offset": "14:00", "glucose": 112.0, "heart_rate": 70.0, "steps": 40, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "sedentary"},
            {"time_offset": "14:15", "glucose": 110.0, "heart_rate": 69.0, "steps": 45, "carbs": 0.0, "sleep_hours": 7.5, "sleep_quality": 84.0, "activity": "sedentary"},
        ],
    },
    "dawn_phenomenon": {
        "title": "Morning Dawn Phenomenon with Poor Rest",
        "description": (
            "Early morning cortisol and growth hormone surge causing hepatic gluconeogenesis. "
            "Coupled with fragmented sleep (4.4 hrs), fasting glucose drifts upward. "
            "Twin forecasts Moderate Risk (45-60%)."
        ),
        "expected_risk": "Moderate (45-60%)",
        "ticks": [
            {"time_offset": "05:30", "glucose": 124.0, "heart_rate": 62.0, "steps": 0, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "sleeping"},
            {"time_offset": "05:45", "glucose": 129.0, "heart_rate": 64.0, "steps": 0, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "sleeping"},
            {"time_offset": "06:00", "glucose": 136.0, "heart_rate": 67.0, "steps": 10, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "sleeping"},
            {"time_offset": "06:15", "glucose": 144.0, "heart_rate": 71.0, "steps": 65, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "sedentary"},
            {"time_offset": "06:30", "glucose": 152.0, "heart_rate": 75.0, "steps": 120, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "light_active"},
            {"time_offset": "06:45", "glucose": 159.0, "heart_rate": 78.0, "steps": 160, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "light_active"},
            {"time_offset": "07:00", "glucose": 164.0, "heart_rate": 80.0, "steps": 140, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "light_active"},
            {"time_offset": "07:15", "glucose": 167.0, "heart_rate": 79.0, "steps": 95, "carbs": 0.0, "sleep_hours": 4.4, "sleep_quality": 45.0, "activity": "sedentary"},
        ],
    },
}


def get_scenario_ticks(scenario: str = "postprandial_spike", num_steps: int = 8) -> List[Dict[str, Any]]:
    """
    Returns sanitized list of sensor observations for the requested scenario.
    """
    scen = CLINICAL_SCENARIOS.get(scenario, CLINICAL_SCENARIOS["postprandial_spike"])
    raw_ticks = scen["ticks"][:num_steps]

    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")

    formatted = []
    for item in raw_ticks:
        time_str = f"{today} {item['time_offset']}:00"
        formatted.append({
            "timestamp": time_str,
            "glucose": float(item["glucose"]),
            "heart_rate": float(item["heart_rate"]),
            "steps": int(item["steps"]),
            "sleep_hours": float(item["sleep_hours"]),
            "sleep_quality": float(item["sleep_quality"]),
            "carbs_intake": float(item["carbs"]),
            "activity_type": item["activity"],
        })

    return formatted
