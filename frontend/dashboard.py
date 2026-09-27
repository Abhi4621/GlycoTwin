"""
Doctor-Facing Healthcare Digital Twin Dashboard.
Provides real-time clinical monitoring, short-term glucose spike forecasts (< 2 hours),
SHAP-based physiological risk factor explanations, and live scenario simulations.
Built with Streamlit & Plotly. Educational & Research Proof-of-Concept.
"""

import datetime
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Direct engine imports for standalone mode / zero-configuration execution
from simulation.scenarios import CLINICAL_SCENARIOS, get_scenario_ticks
from src.data.loader import load_patient_data
from src.digital_twin.engine import DigitalTwinEngine


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="T2D Digital Twin | Clinical Spikes Forecaster",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 15px;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 10px 14px;
        font-size: 12.5px;
        color: #92400E;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .patient-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }
    .badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-mod {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
@st.cache_resource
def get_cached_engine(patient_id: str = "PT-SYNTH-001") -> DigitalTwinEngine:
    """Instantiates and pre-warms the patient's Digital Twin engine."""
    ehr_dict, stream_df = load_patient_data(patient_id=patient_id)
    engine = DigitalTwinEngine(patient_id=patient_id, ehr_profile=ehr_dict)
    engine.seed_initial_history(stream_df)
    return engine


engine = get_cached_engine("PT-SYNTH-001")

if "simulation_step" not in st.session_state:
    st.session_state.simulation_step = 0
if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = "postprandial_spike"


# --- Header & Medical Disclaimers ---
st.markdown('<div class="main-header">🩺 Type 2 Diabetes Physiological Digital Twin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Happiest Health Digital Twin Challenge 2026 | Predictive Glucose Excursion Forecaster</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>RESEARCH & EDUCATIONAL PROOF-OF-CONCEPT ONLY:</strong>
    This system utilizes 100% synthetic patient and sensor data. It is <strong>NOT</strong> an FDA-cleared diagnostic device,
    does <strong>NOT</strong> replace licensed physician evaluation, and must <strong>NEVER</strong> be used to alter insulin
    dosages or pharmacological treatments.
</div>
""", unsafe_allow_html=True)


# --- Patient Profile Card ---
ehr = engine.ehr_profile
col_p1, col_p2, col_p3, col_p4 = st.columns([1.2, 1, 1, 1.4])

with col_p1:
    st.markdown(f"**Patient:** {ehr.get('name', 'Arthur Pendelton')} *(Fictional)*")
    st.caption(f"ID: `{engine.patient_id}` | {ehr.get('gender', 'Male')}, {ehr.get('age', 58)} yrs")

with col_p2:
    st.markdown(f"**BMI:** `{ehr.get('bmi', 29.5)} kg/m²`")
    st.caption(f"BP: {ehr.get('systolic_bp', 132)}/{ehr.get('diastolic_bp', 84)} mmHg")

with col_p3:
    st.markdown(f"**HbA1c:** `{ehr.get('hba1c', 8.1)}%`")
    st.caption(f"Fasting: {ehr.get('fasting_glucose', 138)} mg/dL")

with col_p4:
    st.markdown(f"**Current Therapy:**")
    st.caption(f"{ehr.get('medications_display', 'Metformin 1000mg BID, Empagliflozin 10mg QD')}")

st.divider()


# --- Sidebar: Simulation Controls ---
with st.sidebar:
    st.header("⚙️ Digital Twin Telemetry")
    st.markdown("Simulate live wearable data streaming into the virtual patient:")

    scenario_key = st.selectbox(
        "Select Clinical Scenario",
        options=list(CLINICAL_SCENARIOS.keys()),
        format_func=lambda k: CLINICAL_SCENARIOS[k]["title"],
    )
    st.session_state.active_scenario = scenario_key

    scen_meta = CLINICAL_SCENARIOS[scenario_key]
    st.info(f"**Context:** {scen_meta['description']}\n\n**Expected Spike Risk:** {scen_meta['expected_risk']}")

    ticks = get_scenario_ticks(scenario_key, num_steps=8)
    total_ticks = len(ticks)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Step +15 Min", use_container_width=True):
            if st.session_state.simulation_step < total_ticks:
                next_tick = ticks[st.session_state.simulation_step]
                next_tick["patient_id"] = engine.patient_id
                engine.process_observation(next_tick)
                st.session_state.simulation_step += 1
                st.rerun()

    with col_btn2:
        if st.button("🔄 Reset Twin", use_container_width=True):
            st.session_state.simulation_step = 0
            # Re-seed initial history
            _, stream_df = load_patient_data(patient_id=engine.patient_id)
            engine.seed_initial_history(stream_df)
            st.rerun()

    st.caption(f"Scenario Progress: Step {st.session_state.simulation_step} / {total_ticks}")
    st.progress(st.session_state.simulation_step / max(1, total_ticks))

    # Fast forward all ticks simulation button
    if st.button("⚡ Run Full Scenario Sequence", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        for idx, tick in enumerate(ticks):
            tick["patient_id"] = engine.patient_id
            engine.process_observation(tick)
            progress_bar.progress((idx + 1) / len(ticks))
            status_text.text(f"Tick {idx+1}/{len(ticks)}: Glucose {tick['glucose']} mg/dL")
            time.sleep(0.4)
        st.session_state.simulation_step = total_ticks
        st.success("Simulation sequence completed!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧬 Twin Engine Diagnostics")
    derived = engine.state.derived_state
    st.write(f"- **Buffer Depth:** {len(engine.state.telemetry_buffer)} readings")
    st.write(f"- **Glucose Velocity:** {derived.get('glucose_velocity_mgdl_per_min', 0.0):+.2f} mg/dL/min")
    st.write(f"- **30-Min Trend:** {derived.get('glucose_delta_30m', 0.0):+.1f} mg/dL")
    st.write(f"- **2-Hour Glycemic CV:** {derived.get('cv_percent', 0.0):.1f}%")


# --- Top KPI Summary Cards ---
latest_obs = engine.state.latest_observation or {}
pred = engine.state.latest_prediction
prob = pred.get("spike_probability_2h", 0.15)
risk_lvl = pred.get("risk_level", "Low")
g_val = latest_obs.get("glucose", 120.0)
hr_val = latest_obs.get("heart_rate", 72.0)
steps_val = latest_obs.get("steps", 0)
sleep_h = latest_obs.get("sleep_hours", 7.0)

# Glucose Velocity Arrow
vel = derived.get("glucose_velocity_mgdl_per_min", 0.0)
if vel > 0.8:
    arrow = "↑↑ (Rapid Rise)"
elif vel > 0.2:
    arrow = "↑ (Rising)"
elif vel < -0.8:
    arrow = "↓↓ (Rapid Fall)"
elif vel < -0.2:
    arrow = "↓ (Falling)"
else:
    arrow = "→ (Stable)"

col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)

with col_k1:
    st.metric(
        label="Current CGM Glucose",
        value=f"{g_val:.1f} mg/dL",
        delta=f"{vel:+.2f} mg/dL/min {arrow}",
        delta_color="inverse" if vel > 0.5 else "normal",
    )

with col_k2:
    st.metric(
        label="Heart Rate",
        value=f"{hr_val:.0f} bpm",
        delta=f"{derived.get('autonomic_hr_elevation', 0.0):+.0f} vs base",
    )

with col_k3:
    st.metric(
        label="Recent Activity (2h)",
        value=f"{derived.get('active_steps_2h', 0):,} steps",
        delta=f"{steps_val} steps in last 15m",
    )

with col_k4:
    st.metric(
        label="Prior Sleep Duration",
        value=f"{sleep_h:.1f} hrs",
        delta="Suboptimal (<6h)" if sleep_h < 6.0 else "Adequate",
        delta_color="inverse" if sleep_h < 6.0 else "normal",
    )

with col_k5:
    badge_class = "badge-high" if risk_lvl == "High" else ("badge-mod" if risk_lvl == "Moderate" else "badge-low")
    st.markdown(f"**Predicted 2h Spike Risk**")
    st.markdown(f"<span class='{badge_class}' style='font-size: 20px;'>{prob*100:.1f}% ({risk_lvl})</span>", unsafe_allow_html=True)
    st.caption("Forecast horizon: next 120 minutes")


# --- Main Charts: CGM Stream, 2h Projected Forecast, HR & Activity ---
st.subheader("📈 Multi-Trace Physiological Timeline & 2-Hour Forecast")

buffer_df = engine.state.get_buffer_dataframe()
if not buffer_df.empty:
    buffer_df["dt"] = pd.to_datetime(buffer_df["timestamp"])
    # Show last 32 readings (8 hours) for clear clinical visualization
    plot_df = buffer_df.tail(32).copy().reset_index(drop=True)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=("Continuous Glucose (CGM) with 2-Hour Forward Projection", "Heart Rate (PPG) & Step Activity")
    )

    # Shaded Euglycemic Target Zone (70 to 140 mg/dL)
    fig.add_hrect(
        y0=70, y1=140,
        fillcolor="rgba(34, 197, 94, 0.12)",
        line_width=0,
        annotation_text="Euglycemic Target Range (70-140 mg/dL)",
        annotation_position="bottom right",
        row=1, col=1,
    )

    # Hyperglycemia Excursion Line (>= 180 mg/dL)
    fig.add_hline(
        y=180,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Hyperglycemic Spike Threshold (180 mg/dL)",
        annotation_position="top right",
        row=1, col=1,
    )

    # 1. Historical Glucose Trace
    fig.add_trace(
        go.Scatter(
            x=plot_df["dt"],
            y=plot_df["glucose"],
            mode="lines+markers",
            name="Observed CGM",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=5, color="#1D4ED8"),
        ),
        row=1, col=1,
    )

    # 2. Forward 2-Hour Projected Trajectory
    pred_traj = pred.get("predicted_trajectory", [])
    conf_low = pred.get("confidence_lower", [])
    conf_high = pred.get("confidence_upper", [])

    if pred_traj and len(plot_df) > 0:
        last_dt = plot_df["dt"].iloc[-1]
        future_dts = [last_dt + datetime.timedelta(minutes=15 * s) for s in range(1, len(pred_traj) + 1)]
        # Connect last observed to first projected
        proj_x = [last_dt] + future_dts
        proj_y = [plot_df["glucose"].iloc[-1]] + pred_traj
        proj_low = [plot_df["glucose"].iloc[-1]] + conf_low
        proj_high = [plot_df["glucose"].iloc[-1]] + conf_high

        # Confidence Ribbon
        fig.add_trace(
            go.Scatter(
                x=proj_x + proj_x[::-1],
                y=proj_high + proj_low[::-1],
                fill="toself",
                fillcolor="rgba(249, 115, 22, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Forecast Confidence (±1σ)",
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Forecast Line
        traj_color = "#DC2626" if risk_lvl == "High" else ("#F59E0B" if risk_lvl == "Moderate" else "#10B981")
        fig.add_trace(
            go.Scatter(
                x=proj_x,
                y=proj_y,
                mode="lines+markers",
                name="Digital Twin Forecast",
                line=dict(color=traj_color, width=3, dash="dot"),
                marker=dict(symbol="diamond", size=7, color=traj_color),
            ),
            row=1, col=1,
        )

    # 3. Heart Rate on row 2
    fig.add_trace(
        go.Scatter(
            x=plot_df["dt"],
            y=plot_df["heart_rate"],
            mode="lines",
            name="Heart Rate (bpm)",
            line=dict(color="#E11D48", width=2),
        ),
        row=2, col=1,
    )

    # 4. Steps Bar Chart on row 2 (secondary axis)
    fig.add_trace(
        go.Bar(
            x=plot_df["dt"],
            y=plot_df["steps"],
            name="15m Step Count",
            marker=dict(color="rgba(100, 116, 139, 0.4)"),
            yaxis="y3",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Glucose (mg/dL)", range=[60, max(230, max(plot_df['glucose']) + 20)], row=1, col=1)
    fig.update_yaxes(title_text="HR (bpm)", range=[45, 140], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# --- Section: Explainability & Primary Risk Factors ---
st.subheader("🔍 ML Model Explainability & Clinical Risk Drivers")

col_exp1, col_exp2 = st.columns([1.2, 1])

risk_factors = pred.get("risk_factors", [])

with col_exp1:
    st.markdown("**Top Physiological Risk Drivers (SHAP Local Attribution)**")
    if risk_factors:
        df_rf = pd.DataFrame(risk_factors)
        bar_colors = ["#EF4444" if d == "increases_risk" else "#10B981" for d in df_rf["direction"]]

        fig_bar = go.Figure(go.Bar(
            x=df_rf["attribution_value"],
            y=df_rf["label"],
            orientation="h",
            marker=dict(color=bar_colors),
            text=df_rf["contribution"],
            textposition="auto",
        ))
        fig_bar.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=260,
            xaxis_title="SHAP Attribution Value (Impact on Log-Odds of Spike)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.write("Initializing model feature attributions...")

with col_exp2:
    st.markdown("**Physician Clinical Insights**")
    if risk_factors:
        for item in risk_factors:
            sign_emoji = "⚠️" if item["direction"] == "increases_risk" else "🛡️"
            impact_desc = "drives spike risk upward" if item["direction"] == "increases_risk" else "protective factor against spike"
            st.markdown(
                f"- {sign_emoji} **{item['label']}**: `{item['contribution']}` attribution — *{impact_desc}*."
            )
    st.caption(
        "Attributions are computed directly from the trained tree model via TreeSHAP. "
        "They represent exact quantitative contributions of features to the current 2-hour forecast."
    )


# --- Section: Daily Digital Twin Evolution Timeline ---
st.subheader("⏱️ Today's Digital Twin State Evolution")
history_snapshots = engine.state.state_history_snapshots

if history_snapshots:
    df_snap = pd.DataFrame(history_snapshots).tail(12)
    df_snap["Spike Risk %"] = (df_snap["spike_probability_2h"] * 100).round(1).astype(str) + "%"
    df_snap["Glucose (mg/dL)"] = df_snap["glucose"].round(1)
    df_snap["Heart Rate"] = df_snap["heart_rate"].round(1)
    df_snap["Steps"] = df_snap["steps"]
    df_snap["Velocity (mg/dL/min)"] = df_snap["glucose_velocity"].round(2)
    display_cols = ["timestamp", "Glucose (mg/dL)", "Velocity (mg/dL/min)", "Heart Rate", "Steps", "Spike Risk %", "risk_level"]
    st.dataframe(df_snap[display_cols].rename(columns={"risk_level": "Risk Tier"}), use_container_width=True)
else:
    st.caption("No timeline history recorded yet.")
