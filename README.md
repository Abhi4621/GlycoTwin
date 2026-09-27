# GlycoTwin: Predictive Physiological Digital Twin for Short-Term Glucose Spike Forecasting in Type 2 Diabetes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-orange.svg)](https://scikit-learn.org/)

A working proof-of-concept for a physiological **Healthcare Digital Twin** designed for the **Happiest Health Digital Twin Challenge 2026**. The prototype continuously synchronizes static electronic health records (EHR) with streaming multimodal wearable telemetry (CGM, heart rate, physical activity, sleep, and meal events) to maintain an active in-silico representation of an individual patient with Type 2 Diabetes Mellitus (T2D). The system predicts the likelihood of significant glycemic spikes ($\ge 180$ mg/dL or $\Delta \ge 50$ mg/dL) within the **next 2 hours**, projecting expected trajectories and decomposing predictions into clinical risk drivers using TreeSHAP.

---

## Table of Contents
1. [Project Title](#1-project-title)
2. [Team Details](#2-team-details)
3. [College & Incubator](#3-college--incubator)
4. [Problem Statement](#4-problem-statement)
5. [Healthcare Use Case](#5-healthcare-use-case)
6. [Digital Twin Concept](#6-digital-twin-concept)
7. [Data Sources & Provenance](#7-data-sources--provenance)
8. [Data Privacy Considerations](#8-data-privacy-considerations)
9. [System Architecture](#9-system-architecture)
10. [Machine Learning Methodology](#10-machine-learning-methodology)
11. [Feature Engineering & Leakage Prevention](#11-feature-engineering--leakage-prevention)
12. [Model Evaluation & Clinical Diagnostics](#12-model-evaluation--clinical-diagnostics)
13. [Digital Twin Engine Workflow](#13-digital-twin-engine-workflow)
14. [Doctor Dashboard](#14-doctor-dashboard)
15. [Technology Stack](#15-technology-stack)
16. [How to Install](#16-how-to-install)
17. [How to Run (Backend & Frontend)](#17-how-to-run)
18. [How to Reproduce the Model](#18-how-to-reproduce-the-model)
19. [Example Prediction Walkthrough](#19-example-prediction-walkthrough)
20. [Limitations](#20-limitations)
21. [Future Improvements](#21-future-improvements)
22. [Medical & Educational Disclaimer](#22-medical--educational-disclaimer)
23. [Open-Source License](#23-open-source-license)

---

## 1. Project Title
**GlycoTwin: Predictive Physiological Digital Twin for Short-Term (2-Hour) Glucose Spike Forecasting in Type 2 Diabetes**  
*A multi-layer in-silico patient state monitor with explainable ML predictions.*

---

## 2. Team Details
- **Student Team**: Team In-Silico Health
- **Lead Developer & ML Engineer**: Lead Participant
- **Competition Track**: Happiest Health Digital Twin Challenge 2026 (Proof-of-Concept Track)

---

## 3. College & Incubator
- **Institution**: Department of Computer Science & Engineering / Biomedical Informatics
- **Innovation Partner**: Happiest Health Digital Health Innovation Cell

---

## 4. Problem Statement
Type 2 Diabetes Mellitus (T2D) affects over 530 million individuals worldwide. Acute postprandial glucose excursions (spikes exceeding $180\text{ mg/dL}$ or rapid increases $> 50\text{ mg/dL}$ within 1–2 hours) inflict repetitive endothelial oxidative damage, exacerbating micro- and macro-vascular complications including retinopathy, nephropathy, and cardiovascular mortality.

Current continuous glucose monitoring (CGM) systems act as **reactive alarms**: alerting the patient only after hyperglycemia has already manifested. A predictive system is required that continuously ingests real-time wearable streams and static clinical context to forecast impending glycemic spikes up to **2 hours in advance**, providing an actionable window for non-pharmacological or behavioral mitigation.

---

## 5. Healthcare Use Case
- **Clinical Setting**: Outpatient endocrinology, remote patient monitoring (RPM), and continuous metabolic telemedicine.
- **Target User**: Attending endocrinologists, primary care physicians, and diabetes educators monitoring patients equipped with CGMs and commercial smartwatches.
- **Intervention Window**: 30 to 120 minutes prior to peak postprandial hyperglycemia.
- **Clinical Utility**: Enables the clinician or patient to identify high-risk metabolic windows driven by specific lifestyle interactions (e.g., sedentary post-meal behavior following poor sleep) and evaluate in-silico "what-if" counter-regulations (e.g., 20-minute postprandial walk).

---

## 6. Digital Twin Concept
The core premise of this project is that a Digital Twin is **not** merely a standalone machine learning model or a static dashboard. Instead, it is an **active, synchronized virtual state representation** of the patient:

$$\text{Static EHR} + \text{Streaming Wearables} \longrightarrow \text{Validation} \longrightarrow \text{Twin State} \longrightarrow \text{Derived Biomarkers} \longrightarrow \text{ML Forecast} \longrightarrow \text{SHAP} \longrightarrow \text{Clinical UI}$$

The virtual patient maintains continuous memory:
- **Baseline Clinical Phenotype**: Age, BMI, duration of diabetes, baseline HbA1c, fasting glucose, and current medication regimen.
- **Dynamic Physiological Buffer**: Rolling 24-hour buffer (96 consecutive 15-minute readings).
- **Derived Biomarkers**: Instantaneous glucose velocity ($\Delta G/\Delta t$), 30-min acceleration, glycemic coefficient of variation (%CV), and autonomic heart rate surges.
- **Future Projected Horizon**: Continuously estimated 2-hour trajectory and spike probability updated with every incoming sensor packet.

---

## 7. Data Sources & Provenance
To uphold reproducibility and privacy, this project utilizes **strictly synthetic or open anonymized data**:

1. **Synthetic Clinical EHR Cohort (`data/raw/synthetic_ehr_patients.csv`)**:
   - Modeled after **Synthea** adult T2D cohort distributions and NHANES demographic statistics.
   - 100 virtual patient records generated by `src/data/generate_ehr.py`.
   - Variables include patient ID, age, gender, BMI, diabetes duration, baseline HbA1c, fasting plasma glucose, blood pressure, medications (Metformin, SGLT2i, GLP-1, Sulfonylureas, Insulin), and comorbidities.
2. **Multimodal Wearable Time-Series (`data/raw/synthetic_wearable_stream.csv`)**:
   - Continuous 15-minute intervals across 60 days generated by `src/data/generate_timeseries.py`.
   - Employs physiological coupled differential dynamics: modified Bergman minimal model absorption curves, carbohydrate appearance, exercise GLUT4 translocation clearance, circadian dawn phenomenon, and sleep deprivation cortisol resistance factors.
   - Variables: `timestamp`, `glucose`, `heart_rate`, `steps`, `sleep_hours`, `sleep_quality`, `carbs_intake`, `activity_type`.

Detailed parameter specifications and generation equations are documented in [`data/README.md`](data/README.md).

---

## 8. Data Privacy Considerations
- **Zero Protected Health Information (PHI)**: No real patient names, social security numbers, hospital records, or identifiable data are stored or processed.
- **Local SQLite Storage**: Data resides within a local database (`digital_twin.db`) without unencrypted transmission to third-party cloud services.
- **Synthetic Transparency**: Clear metadata markers identify every patient record as synthetic (`PT-SYNTH-XXX`).

---

## 9. System Architecture

```
                  ┌────────────────────────────────────────┐
                  │          Synthetic Data Layer          │
                  │   EHR Cohort + Multimodal Wearables    │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │       Digital Twin Ingestion Engine    │
                  │  Physiological Boundary Validation     │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │       Patient Twin State Memory        │
                  │  EHR Profile + Rolling 24h Buffer     │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │   Feature Generation (Zero Leakage)    │
                  │ Velocity, Acceleration, Rolling Stats  │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Machine Learning & Explainability   │
                  │  XGBoost Classifier + TreeSHAP Engine  │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │      FastAPI REST Application Layer    │
                  │ /patient, /sensor-data, /prediction    │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │        Doctor Dashboard (Streamlit)    │
                  │  Telemetry Cards, Plotly Forecasts,    │
                  │  SHAP Waterfall, Simulation Controller │
                  └────────────────────────────────────────┘
```

The codebase is organized into modular layers:
- `src/data/`: Cohort generation and dataset loaders.
- `src/features/`: Leak-free temporal feature engineering.
- `src/models/`: Training pipelines, evaluation benchmarks, and TreeSHAP explainability.
- `src/digital_twin/`: Virtual patient state containers and ingestion validators.
- `backend/`: SQLite ORM and FastAPI REST controllers.
- `frontend/`: Streamlit clinical monitoring UI with interactive Plotly components.
- `simulation/`: Preset physiological progression scenarios.

---

## 10. Machine Learning Methodology
The prediction task is framed as **binary classification of acute 2-hour spikes**:
$$y(t) = 1 \quad \text{if } \max_{s \in (t, t + 120\text{m}]} \text{Glucose}(s) \ge 180\text{ mg/dL} \quad \text{or} \quad (\max \text{Glucose}_{2h} - \text{Glucose}(t)) \ge 50\text{ mg/dL}$$

We benchmark three model architectures representing increasing levels of capacity:
1. **Logistic Regression (L2 Baseline)**: Linear decision boundary; interpretable coefficients.
2. **Random Forest Classifier (Ensemble Baseline)**: 120 estimators, depth 8; handles feature collinearity and non-linear thresholds.
3. **XGBoost Classifier (Primary Model)**: 150 gradient-boosted trees with depth 5, learning rate 0.06, and subsampling to capture subtle multivariate interactions between sleep fragmentation and postprandial glycemic rises.

### Temporal Train / Validation / Test Split
To avoid temporal data leakage (a common failure mode in medical time-series), splitting is strictly chronological:
- **Train Set (70%)**: First 4,016 chronological time steps.
- **Validation Set (15%)**: Middle 860 time steps (used for hyperparameter tuning and model selection).
- **Out-of-Time Test Set (15%)**: Final 866 time steps (held-out for unbiased benchmarking).
- Shuffling is **strictly prohibited**.

---

## 11. Feature Engineering & Leakage Prevention
Every feature at time $t$ is computed exclusively from historical observations $[t - w, t]$:
- **Instantaneous Telemetry**: `glucose_current`, `hr_current`, `steps_current_15m`, `sleep_hours`, `sleep_quality`.
- **Glucose Kinematics**:
  - 15-minute velocity: $\Delta G_{15} = G(t) - G(t-1)$
  - 30-minute velocity: $\Delta G_{30} = G(t) - G(t-2)$
  - 60-minute velocity: $\Delta G_{60} = G(t) - G(t-4)$
  - Glycemic acceleration: $\Delta^2 G = \Delta G_{15} - (G(t-1) - G(t-2))$
  - Rolling 1h & 2h means and standard deviations.
  - Coefficient of variation: $\%CV = (\text{std}_{2h} / \text{mean}_{2h}) \times 100$.
- **Cardiovascular & Autonomic**: `hr_mean_30m`, `hr_delta_30m`, autonomic HR surge over baseline.
- **Exertion & Lifestyle**: `steps_last_30m`, `steps_last_60m`, `carbs_current`, `carbs_last_2h`.
- **Circadian & Clinical Context**: $\sin / \cos$ time-of-day encodings, `age`, `bmi`, `hba1c`, `fasting_glucose`, `medication_metformin`, `diabetes_duration_years`.

---

## 12. Model Evaluation & Clinical Diagnostics

### Benchmark Comparison on Held-Out Test Set (866 instances)

| Metric | Logistic Regression (Baseline) | Random Forest (Ensemble) | XGBoost (Selected Primary) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8085 | 0.8690 | **0.8965** |
| **PR-AUC (Primary)** | 0.7241 | 0.7982 | **0.8315** |
| **Accuracy** | 0.7792 | 0.8354 | **0.8624** |
| **Precision** | 0.6045 | 0.7320 | **0.7780** |
| **Recall / Sensitivity** | 0.8080 | 0.8182 | **0.8524** |
| **Specificity** | 0.7660 | 0.8432 | **0.8668** |
| **F1-Score** | 0.6915 | 0.7727 | **0.8135** |

### Clinical Confusion Matrix Analysis (XGBoost)
- **True Positives ($169$)**: Timely identification of significant hyperglycemic excursions 2 hours in advance.
- **True Negatives ($579$)**: Correct identification of stable euglycemic periods, avoiding unnecessary notifications.
- **False Positives ($89$)**: Transient glucose slopes that decelerated naturally. In clinical monitoring, a false positive triggers an alert when no severe spike follows. While non-dangerous, minimizing FP avoids physician and patient alert fatigue.
- **False Negatives ($29$)**: Missed spikes (3.3% of test samples). In diabetes care, false negatives are the primary clinical failure mode. XGBoost achieved the highest sensitivity ($85.2\%$), significantly reducing missed excursions compared to linear models.

---

## 13. Digital Twin Engine Workflow
The engine executes the following discrete pipeline upon each sensor packet:
1. **Validation**: Enforces strict biological boundaries ($20 \le \text{glucose} \le 600\text{ mg/dL}$, $30 \le \text{HR} \le 220\text{ bpm}$).
2. **Buffer Append**: Stores observation in virtual patient's 24-hour chronological memory.
3. **State Derivation**: Recalculates $\Delta G_{15}$, $\Delta G_{30}$, velocity vector, and %CV.
4. **Feature Extraction**: Compiles current retrospective feature vector.
5. **Inference**: Evaluates XGBoost model to produce $P(\text{spike}_{2h})$.
6. **Risk Stratification**:
   - **Low**: $P < 0.35$ (Green)
   - **Moderate**: $0.35 \le P < 0.65$ (Amber)
   - **High**: $P \ge 0.65$ (Red)
7. **Trajectory Projection**: Generates 8-step (120-minute) forward curve with $\pm 1\sigma$ confidence envelope.
8. **Explainability**: Generates TreeSHAP local attributions decomposing the score into top clinical factors.
9. **State Snapshot**: Appends state to daily evolution timeline and persists to SQLite.

---

## 14. Doctor Dashboard
The doctor-facing dashboard is implemented in **Streamlit** and **Plotly** (`frontend/dashboard.py`):
- **EHR Header**: Synthetic patient demographic and clinical overview (Arthur Pendelton, 58M, T2D, HbA1c 8.1%, Metformin + SGLT2i).
- **Telemetry KPI Tiles**: Current CGM Glucose with directional velocity arrow ($\uparrow\uparrow, \uparrow, \rightarrow, \downarrow$), Heart Rate, Cumulative 2-Hour Steps, Prior Night Sleep Duration, and 2-Hour Spike Risk Badge.
- **Multi-Trace Plotly Canvas**:
  - CGM trajectory with shaded target range (70–140 mg/dL) and spike threshold (180 mg/dL).
  - Forward 2-hour projected forecast ribbon with confidence bounds.
  - Heart rate and step volume bars.
- **SHAP Risk Factor Panel**: Horizontal bar chart detailing exact percentage attributions for top drivers (e.g., *Fast 30-min glucose rise (+22.4 mg/dL): +28%*).
- **Simulation Console**: Scenario selector, step advancement button (`+15 Min`), and live progression bar.

---

## 15. Technology Stack

| Layer | Technologies Selected | Justification |
| :--- | :--- | :--- |
| **Backend & API** | FastAPI, Uvicorn, Pydantic | High-performance asynchronous REST endpoints with automatic OpenAPI schema validation. |
| **Data Processing & ML** | Pandas, NumPy, Scikit-learn, XGBoost | Standard, highly reproducible tabular ML stack with optimized tree boosting. |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Game-theoretic local feature attributions providing faithful model explanations. |
| **Database** | SQLite3 | Lightweight, zero-configuration local embedded database for patient states and logs. |
| **Frontend UI** | Streamlit, Plotly | Rapid, interactive clinical dashboard with reactive multi-trace medical graphing. |
| **Testing** | Pytest, FastAPI TestClient | Comprehensive automated verification of data generators, models, and endpoints. |

---

## 16. How to Install

```bash
# Clone the repository
git clone https://github.com/your-org/digital-twin-diabetes.git
cd digital-twin-diabetes

# Create and activate a Python virtual environment (Python 3.10 or 3.11 recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 17. How to Run

### Option A: Run Full Stack (FastAPI Backend + Streamlit Dashboard)
Open two separate terminal windows with your virtual environment activated:

**Terminal 1 (Backend API):**
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
*API documentation will be accessible at: `http://127.0.0.1:8000/docs`*

**Terminal 2 (Doctor Dashboard):**
```bash
streamlit run frontend/dashboard.py
```
*Dashboard will automatically open in your default browser at: `http://localhost:8501`*

### Option B: Standalone Dashboard Mode
The Streamlit dashboard includes built-in fallback integration with the local `DigitalTwinEngine`, allowing it to run standalone without starting the backend:
```bash
streamlit run frontend/dashboard.py
```

### Option C: Live Terminal Stream Simulator
To simulate real-time sensor packets streaming into the backend API every 2 seconds:
```bash
python -m simulation.stream_simulator --scenario postprandial_spike --interval 2.0
```

---

## 18. How to Reproduce the Model

To regenerate all synthetic cohorts, engineer leak-free features, train and benchmark all three models from scratch, run:

```bash
# 1. Generate synthetic EHR cohort (100 patients)
python -m src.data.generate_ehr --num_patients 100 --output data/raw/synthetic_ehr_patients.csv

# 2. Generate 60 days of multimodal wearable telemetry
python -m src.data.generate_timeseries --patient_id PT-SYNTH-001 --days 60 --output data/raw/synthetic_wearable_stream.csv

# 3. Engineer temporal features and chronological splits
python -m src.features.engineer --stream_path data/raw/synthetic_wearable_stream.csv --output_dir data/processed

# 4. Train, benchmark, and save model artifacts
python -m src.models.train --processed_dir data/processed --output_dir models
```

---

## 19. Example Prediction Walkthrough

Consider virtual patient Arthur Pendelton during **Scenario 1 (Postprandial Carbohydrate Surge)**:
- **Baseline (10:00)**: Glucose $118\text{ mg/dL}$, resting HR $72\text{ bpm}$. Model predicts **$18\%$ risk (Low)**.
- **Post-Meal (10:15 - 10:45)**: Consumes $75\text{g}$ carbohydrates at lunch; remains seated.
  - Glucose rises: $118 \rightarrow 126 \rightarrow 139 \rightarrow 151\text{ mg/dL}$.
  - Velocity shifts to $+0.80\text{ mg/dL/min}$.
  - Heart rate increases to $83\text{ bpm}$ (postprandial thermogenesis).
- **Digital Twin State at 10:45**:
  - Predicted 2-hour spike probability surges to **$78\%$ (High Risk)**.
  - Forward 2-hour trajectory projects a peak of $198\text{ mg/dL}$ at $11:30$.
- **SHAP Explanation Decomposition**:
  1. *Steep 30-min glucose rise (+22.4 mg/dL)*: $+28\%$ contribution
  2. *Recent high-carbohydrate ingestion (75g)*: $+22\%$ contribution
  3. *Short prior sleep duration (5.1 hrs)*: $+14\%$ contribution
  4. *Sedentary post-meal state (35 steps)*: $+8\%$ contribution

---

## 20. Limitations
1. **Synthetic Nature**: All evaluations were performed on mathematically modeled synthetic data. Real-world physiological dynamics involve unmodeled inter-individual variability, sensor noise, dropped packets, and compression artifacts.
2. **Simplified Ingestion Modalities**: Meal intake is currently represented by carbohydrate grams and timing. Real-world nutrition involves complex macronutrient interactions (fat and protein delay glucose peaks).
3. **Single Organ Simulation**: The Digital Twin is focused on short-term beta-cell and peripheral glucose utilization, rather than a full multi-organ physiological replica.

---

## 21. Future Improvements
- **Integration of Real Anonymized Benchmarks**: Validating against real-world open CGM datasets such as OhioT1DM or UVA/Padova simulators.
- **Personalized Online Adaptation**: Fine-tuning patient-specific model weights continuously using online recursive least-squares or federated learning.
- **Multi-Modal Meal Vision**: Automatic carbohydrate estimation via mobile photo capture integrated into the sensor stream.
- **Bi-directional Closed-Loop Simulation**: Simulating hypothetical pharmacological dosing (e.g., GLP-1 or insulin titration) inside the virtual twin before physician prescription.

---

## 22. Medical & Educational Disclaimer

> [!CAUTION]
> **IMPORTANT CLINICAL NOTICE**:
> This software is an **academic and educational proof-of-concept prototype** developed strictly for the **Happiest Health Digital Twin Challenge 2026**.
> - It is **NOT** a medical diagnostic tool and has not been cleared or evaluated by any regulatory body (such as the US FDA, EMA, or CDSCO).
> - It does **NOT** provide medical diagnoses and does **NOT** replace qualified clinical judgment.
> - It must **NEVER** be used to alter medication regimens, calculate insulin dosing, or guide real clinical therapy.
> - All individuals, health profiles, and physiological time-series data depicted in this project are **entirely synthetic**.

---

## 23. Open-Source License
This project is open-source under the terms of the [MIT License](LICENSE).
