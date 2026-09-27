# Presentation Guide & Video Demonstration Script
### Happiest Health Digital Twin Challenge 2026
**Topic**: Type 2 Diabetes Physiological Digital Twin for Short-Term Glucose Spike Prediction (< 2 Hours)
**Duration**: 20 Minutes

---

## 1. Presentation Structure Overview (20 Minutes)

| Segment | Timing | Core Focus | Visual / Screen |
| :--- | :--- | :--- | :--- |
| **1. Introduction & Clinical Need** | 00:00 – 03:00 | Why reactive CGM alarms are insufficient; Type 2 Diabetes postprandial excursions. | Problem statement & clinical motivation slides. |
| **2. The Digital Twin Concept** | 03:00 – 06:00 | Fusion of static clinical EHR + dynamic multimodal wearables into an active in-silico twin. | Architecture flow diagram. |
| **3. Machine Learning & Zero-Leakage Pipeline** | 06:00 – 10:00 | Chronological train/val/test split, feature engineering, LR vs RF vs XGBoost comparison, and SHAP. | Benchmark table & PR-AUC curves. |
| **4. Live Interactive Twin Demonstration** | 10:00 – 16:30 | **Centerpiece Demo**: Normal baseline $\rightarrow$ Post-meal carbohydrate surge $\rightarrow$ Real-time state update $\rightarrow$ Spike prediction $\rightarrow$ SHAP explanation. | Streamlit Doctor Dashboard + Live Simulation. |
| **5. Euglycemic Control Counter-Scenario** | 16:30 – 18:30 | Showing how the Digital Twin reflects physical activity blunting glucose excursions. | Euglycemic Walk scenario. |
| **6. Limitations, Disclaimers & Future Work** | 18:30 – 20:00 | Academic research disclaimer, synthetic data limitations, and path toward multi-organ twins. | Final takeaway slide. |

---

## 2. Minute-by-Minute Demonstration Script

### Minute 00:00 – 03:00: Problem Statement & Healthcare Challenge
- **Speaker**: "Hello judges and evaluators of the Happiest Health Digital Twin Challenge 2026. Today, we are presenting our working proof-of-concept for a Healthcare Digital Twin focused on Type 2 Diabetes."
- **Key point**: "Over 500 million people live with Type 2 Diabetes. The greatest contributor to diabetic cardiovascular disease and microvascular complications is acute postprandial glycemic excursions—spikes over 180 mg/dL. Existing CGMs are reactive alarms: they beep after the patient is already hyperglycemic. Our goal is predictive: forecast spikes up to 2 hours before they occur, giving patients and clinicians actionable warning."

### Minute 03:00 – 06:00: The Digital Twin Architecture
- **Speaker**: "A Digital Twin is not just an ML model or a dashboard. It is an active physiological state loop:
  1. Static EHR defines the baseline patient phenotype: age, BMI, baseline HbA1c, fasting glucose, and current medications.
  2. Dynamic wearable streams arrive every 15 minutes: CGM, optical heart rate, step count, and sleep metrics.
  3. The Ingestion Engine validates the telemetry against physiological laws.
  4. The Patient Twin State continually recalculates derived biomarkers: glucose velocity, 30-minute acceleration, and autonomic stress elevation."

### Minute 06:00 – 10:00: Machine Learning Methodology & Rigorous Evaluation
- **Speaker**: "We strictly avoided time-series data leakage:
  - Splitting was 100% chronological: 70% past for training, 15% validation for hyperparameter tuning, 15% out-of-time test set. No future data leaks into the past.
  - We compared three distinct architectures:
    1. Logistic Regression: Transparent linear baseline (PR-AUC 0.724).
    2. Random Forest: Nonlinear tree ensemble (PR-AUC 0.798).
    3. XGBoost: Gradient boosted trees (PR-AUC 0.832, ROC-AUC 0.897, Recall 85.2%).
  - XGBoost was selected as our deployed model because in clinical monitoring, high sensitivity (low false negatives) is vital to catch acute spikes before they cause harm."

### Minute 10:00 – 16:30: Live Demonstration of the Digital Twin Loop (The Centerpiece)
- **Speaker**: "Now, let us switch to our live Doctor Dashboard running on virtual patient Arthur Pendelton, a 58-year-old male with Type 2 Diabetes, BMI 29.5, and HbA1c 8.1%."
- **Action**: Show Arthur's baseline:
  - Glucose: 118 mg/dL (Euglycemic).
  - Risk Level: **Low (18%)**.
- **Action**: Trigger Scenario 1: Postprandial Glucose Surge.
  - Step 1 (10:15): Arthur finishes a 75g carb lunch. Glucose shifts to 126 mg/dL. Twin velocity increases to $+0.53$ mg/dL/min.
  - Step 2 (10:30): Glucose hits 139 mg/dL. Heart rate rises due to thermogenesis. Risk crosses into **Moderate (52%)**.
  - Step 3 (10:45): Glucose reaches 151 mg/dL. Upward acceleration detected. Twin triggers **High Predicted Spike Risk (78%)**.
  - Step 4 (11:00): Glucose climbs to 168 mg/dL. The 2-hour projected forecast ribbon shows expected peak reaching 206 mg/dL.
- **Action**: Highlight the Explainability Panel:
  - Point to the TreeSHAP attribution waterfall:
    - $+28\%$ contribution from steep 30-min glucose velocity.
    - $+22\%$ contribution from recent carbohydrate ingestion.
    - $+14\%$ contribution from prior night sleep deprivation (5.1 hrs).
    - $+8\%$ from post-meal sedentary behavior.

### Minute 16:30 – 18:30: Counter-Scenario (Euglycemic Control via Physical Activity)
- **Speaker**: "Now let's demonstrate physiological counter-regulation. In Scenario 2, Arthur consumes a moderate meal but immediately takes an active 20-minute walk (1,420 steps)."
- **Action**: Step through Scenario 2:
  - Glucose blunts at 126 mg/dL and promptly settles back to 112 mg/dL.
  - The Digital Twin factors in muscle GLUT4 uptake from steps, keeping predicted risk securely in the **Low range (< 20%)**.

### Minute 18:30 – 20:00: Academic Disclaimers, Limitations & Conclusion
- **Speaker**: "To conclude:
  - This system is an educational and research proof-of-concept using 100% synthetic data.
  - It does not replace medical consultation or automate medication adjustments.
  - By providing a predictive physiological window, our Type 2 Diabetes Digital Twin demonstrates the transformative potential of proactive, personalized virtual patient care."
