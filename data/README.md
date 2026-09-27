# Dataset Documentation & Transparency Statement

## 1. Overview
This directory contains raw and processed data used by the **Type 2 Diabetes Healthcare Digital Twin (Proof-of-Concept)** developed for the **Happiest Health Digital Twin Challenge 2026**.

```
data/
├── raw/
│   ├── synthetic_ehr_patients.csv       # Static EHR records for 100 virtual T2D patients
│   └── synthetic_wearable_stream.csv    # 15-minute multimodal time-series (CGM, HR, steps, sleep, meals)
├── processed/
│   ├── train_features.csv               # 70% chronological training split
│   ├── val_features.csv                 # 15% validation split
│   ├── test_features.csv                # 15% out-of-time test split
│   └── scaler.joblib                    # Robust feature scaling pipeline
└── README.md                            # Data specification & provenance
```

---

## 2. Data Sources & Provenance

### 2.1 Synthetic Clinical EHR (`synthetic_ehr_patients.csv`)
- **Origin**: Synthetically generated using a clinical cohort engine modeled after **Synthea** distributions for adult patients diagnosed with Type 2 Diabetes Mellitus (T2D).
- **Generator script**: `src/data/generate_ehr.py`
- **Cohort Size**: 100 virtual patients.
- **Attributes**:
  - `patient_id`: Unique identifier (e.g., `PT-SYNTH-001`).
  - `name`: Fictional synthetic name.
  - `age`: 35–78 years (Gaussian distributed around 58 ± 10).
  - `gender`: Female / Male (50/50 balanced).
  - `bmi`: 22.0–42.0 kg/m² (Log-normal distribution reflecting T2D obesity demographics, median 29.5).
  - `diabetes_duration_years`: 1–25 years.
  - `hba1c`: 6.5%–11.5% (Baseline glycemic control).
  - `fasting_glucose`: 110–220 mg/dL.
  - `systolic_bp` / `diastolic_bp`: Blood pressure measurements (mmHg).
  - `medications`: Standard T2D therapies (Metformin, SGLT-2 inhibitors, GLP-1 receptor agonists, Sulfonylureas, Basal Insulin).
  - `comorbidities`: Hypertension, Dyslipidemia, Retinopathy history.
  - `family_history_diabetes`: Binary (0 or 1).

### 2.2 Multimodal Wearable & CGM Stream (`synthetic_wearable_stream.csv`)
- **Origin**: High-fidelity continuous time-series simulator capturing physiological glucose-insulin-lifestyle dynamics.
- **Generator script**: `src/data/generate_timeseries.py`
- **Sampling Frequency**: Uniform 15-minute intervals ($4 \times 24 = 96$ ticks per day).
- **Time Window**: 30 to 90 continuous days per monitored patient.
- **Biomarkers & Telemetry**:
  - `timestamp`: ISO-8601 formatted datetime (`YYYY-MM-DD HH:MM:SS`).
  - `glucose`: Interstitial Continuous Glucose Monitoring (CGM) reading in mg/dL.
  - `heart_rate`: Optical photoplethysmography (PPG) heart rate in beats per minute (bpm).
  - `steps`: Pedometer step counts accumulated within the 15-minute interval.
  - `sleep_hours`: Total sleep duration recorded the preceding night.
  - `sleep_quality`: Sleep efficiency score (0–100, incorporating restorative deep/REM sleep).
  - `carbs_intake`: Carbohydrate intake in grams (at breakfast ~08:00, lunch ~13:00, dinner ~19:00, or snacks).
  - `activity_type`: `sedentary`, `light_active`, `moderate_walk`, `strenuous_exercise`, or `sleeping`.

---

## 3. Mathematical & Physiological Dynamics Engine
The synthetic time-series engine does **not** generate arbitrary noise. Instead, it simulates coupled physiological differential dynamics:

1. **Basal Circadian Rhythm & Dawn Phenomenon**:
   $$G_{\text{basal}}(t) = G_0 + A_{\text{dawn}} \cdot \exp\left(-\frac{(t - 6.5)^2}{2 \sigma_{\text{dawn}}^2}\right)$$
   Models the early morning hepatic glucose release driven by cortisol and growth hormone surges.

2. **Postprandial Carbohydrate Absorption**:
   Following a meal at $t_{\text{meal}}$ with carbs $C$, systemic glucose appearance follows a dual-compartment gut absorption curve:
   $$R_a(t) = C \cdot \frac{k_{\text{abs}}^2 (t - t_{\text{meal}})}{\tau_{\max}} \exp\left(-\frac{t - t_{\text{meal}}}{\tau_{\max}}\right) \cdot S_I^{-1}$$
   where $S_I$ is the patient's individual insulin sensitivity inversely scaled by BMI and HbA1c.

3. **Physical Activity Glucose Utilization**:
   Muscular glucose uptake during active steps:
   $$U_{\text{exercise}}(t) = k_{\text{steps}} \cdot \text{steps}(t) \cdot (1 + 0.5 \cdot \text{active\_state})$$

4. **Sleep Disruption & Cortisol Impairment**:
   When sleep duration $< 6.0$ hours or sleep quality $< 65$, baseline insulin resistance is increased by 15–30% for that 24-hour cycle.

5. **Heart Rate Dynamics**:
   Coupled with circadian resting heart rate, step acceleration, and postprandial thermogenesis.

---

## 4. Academic & Research Transparency Declaration

> [!IMPORTANT]
> **Data Privacy and Synthetic Disclosure**:
> - All patient records, identifiers, demographic details, and sensor streams within this project are **100% synthetically generated** or anonymized open benchmarks.
> - **Zero Protected Health Information (PHI)** or real patient identity was collected, stored, or processed.
> - **Synthetic data does not represent actual clinical patient outcomes.** The simulated physiological dynamics are designed specifically for algorithmic development, architecture validation, and user interface demonstration in the **Happiest Health Digital Twin Challenge 2026**.
> - This software must not be used to guide real-world patient therapy or clinical treatment.

---

## 5. How to Regenerate the Datasets
To reproduce all synthetic data and generate the processed training splits, run:
```bash
# 1. Generate synthetic EHR profiles
python -m src.data.generate_ehr --num_patients 100 --output data/raw/synthetic_ehr_patients.csv

# 2. Generate wearable time-series stream
python -m src.data.generate_timeseries --patient_id PT-SYNTH-001 --days 60 --output data/raw/synthetic_wearable_stream.csv

# 3. Build features and chronological train/val/test splits
python -m src.features.engineer --input data/raw/synthetic_wearable_stream.csv --output_dir data/processed/
```
