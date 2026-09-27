"""
Live Real-Time Telemetry Stream Simulator.
Streams sensor observations to the FastAPI backend or directly to the Digital Twin engine.
"""

import argparse
import sys
import time
from typing import Optional
import requests

from simulation.scenarios import CLINICAL_SCENARIOS, get_scenario_ticks


def run_stream_simulation(
    scenario: str = "postprandial_spike",
    api_url: str = "http://127.0.0.1:8000",
    interval_seconds: float = 2.0,
    patient_id: str = "PT-SYNTH-001",
) -> None:
    """
    Sequentially streams sensor ticks to the backend endpoint, observing the Digital Twin respond.
    """
    scen_info = CLINICAL_SCENARIOS.get(scenario, CLINICAL_SCENARIOS["postprandial_spike"])
    print("=" * 75)
    print(f"Starting Digital Twin Live Stream Simulation: {scen_info['title']}")
    print(f"Description: {scen_info['description']}")
    print(f"Target Patient: {patient_id} | Stream Cadence: 1 reading every {interval_seconds}s")
    print(f"Target URL: {api_url}/sensor-data")
    print("=" * 75)

    ticks = get_scenario_ticks(scenario=scenario, num_steps=8)

    for i, tick in enumerate(ticks, start=1):
        tick["patient_id"] = patient_id
        timestamp = tick["timestamp"]
        glucose = tick["glucose"]
        hr = tick["heart_rate"]
        steps = tick["steps"]

        try:
            resp = requests.post(f"{api_url}/sensor-data", json=tick, timeout=5)
            if resp.status_code == 200:
                # Query updated twin prediction
                pred_resp = requests.get(f"{api_url}/patient/{patient_id}/prediction", timeout=5).json()
                prob = pred_resp.get("spike_probability_2h", 0.0)
                risk_lvl = pred_resp.get("risk_level", "Unknown")

                # ASCII Risk Bar
                bar_len = int(prob * 20)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                color_flag = "ALERT!" if risk_lvl == "High" else "OK"

                print(
                    f"[{timestamp}] TICK #{i} -> Glucose: {glucose:5.1f} mg/dL | "
                    f"HR: {hr:4.1f} bpm | Steps: {steps:4d} | "
                    f"Risk: [{bar}] {prob*100:4.1f}% ({risk_lvl}) {color_flag}"
                )
            else:
                print(f"API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Could not reach API at {api_url} ({e}). Emulating local tick output:")
            print(f"[{timestamp}] TICK #{i} -> Glucose: {glucose:5.1f} mg/dL | HR: {hr:4.1f} bpm | Steps: {steps:4d}")

        if i < len(ticks):
            time.sleep(interval_seconds)

    print("=" * 75)
    print(f"Simulation completed! Final state available at {api_url}/patient/{patient_id}/current-state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream sensor data into the Digital Twin.")
    parser.add_argument("--scenario", type=str, default="postprandial_spike", choices=list(CLINICAL_SCENARIOS.keys()))
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between ticks")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--patient_id", type=str, default="PT-SYNTH-001")
    args = parser.parse_args()

    run_stream_simulation(
        scenario=args.scenario,
        api_url=args.api_url,
        interval_seconds=args.interval,
        patient_id=args.patient_id,
    )
