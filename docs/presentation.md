# GlycoTwin Presentation & Pitch Deck Guide
### Happiest Health Digital Twin Challenge 2026

**Official Slide Deck**: [`docs/GlycoTwin_Presentation.pdf`](GlycoTwin_Presentation.pdf)  
**Team**: GlucoStudio • **College**: VIT Bhopal University  
**Team Members**: Abhi Pandey, Tanishq Das  

---

## 1. Slide Deck Overview (10 Slides)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GLYCOTWIN SLIDE DECK                           │
│                                                                        │
│  [Slide 1]  Title: GlycoTwin – Personalized Digital Twin for Glucose   │
│  [Slide 2]  The Problem We Started With: Glucose Response is Personal  │
│  [Slide 3]  Our Approach: GlycoTwin (Collect → Understand → Respond)   │
│  [Slide 4]  From Health Data to a Digital Twin (Static + Dynamic)      │
│  [Slide 5]  System Workflow: 7-Second Physiological Ingestion Loop     │
│  [Slide 6]  Technical Architecture: Full-Stack & AI Pipeline           │
│  [Slide 7]  How We Estimate Glucose Impact: Inputs, Processing, Output │
│  [Slide 8]  From Meal Photo to Insight: Barcode, Vision, Context       │
│  [Slide 9]  Meet the Twin: Interactive Clinical & Patient UI          │
│  [Slide 10] Why Personalization Matters: Person A vs Person B Context  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Slide-by-Slide Breakdown & Presentation Script

### Slide 1: Title Slide
* **Visual**: GlycoTwin Cover Graphic, Medical Twin Illustration.
* **Content**:
  * **Title**: **GlycoTwin**
  * **Subtitle**: *A Personalized Digital Twin for Glucose Management*
  * **Challenge**: Happiest Health Digital Twin Challenge 2026
  * **Team**: GlucoStudio • **College**: VIT Bhopal University
  * **Team Members**: Abhi Pandey, Tanishq Das
* **Speaking Script**:
  > *"Good morning evaluators and judges. We are Team GlucoStudio from VIT Bhopal University, represented by Abhi Pandey and Tanishq Das. Today, we are proud to introduce **GlycoTwin**: a personalized Healthcare Digital Twin for short-term glucose management and predictive spike forecasting in Type 2 Diabetes."*

---

### Slide 2: The Problem We Started With
* **Visual**: Meal intake, sleep disruption, and diverging individual glucose response curves.
* **Content**:
  * Daily decisions are repetitive: *"What will this meal do to my glucose?"*
  * Responses depend on recent activity, sleep, and prior meals — not just the food itself.
  * Generic dietary guidance misses individual physiological patterns and context.
  * **Main Message**: **Glucose response is deeply personal** — the exact same meal affects two individuals in completely different ways.
* **Speaking Script**:
  > *"Every single day, individuals living with Type 2 Diabetes face repeated uncertainty: 'If I eat this lunch, will my blood sugar spike?' Standard nutritional advice gives generic carb counts, but human physiology doesn't work in isolation. The glycemic impact of a meal is dictated by how you slept last night, your physical activity over the past hour, and your baseline insulin sensitivity. GlycoTwin solves this by moving from generic guidelines to an individualized virtual patient model."*

---

### Slide 3: Our Approach: GlycoTwin
* **Visual**: Three-stage progression header + cyclical state flow diagram.
* **Content**:
  * **1. Collect**: Glucose readings, meals/photos, sleep, activity, user profile.
  * **2. Understand**: Combine recent + historical data, identify patterns, estimate physiological impact.
  * **3. Respond**: Risk/impact score, personalized insights, food alternatives, twin interaction.
  * **Cycle**: `User Data` $\rightarrow$ `Personal Health State` $\rightarrow$ `GlycoTwin Stateful Representation` $\rightarrow$ `Prediction + Insights` $\rightarrow$ `User / Clinician`.
* **Speaking Script**:
  > *"Our approach centers on a continuous three-pillar loop: Collect, Understand, and Respond. We collect multi-modal telemetry from wearables and user inputs. We combine recent sensor streams with clinical history to build a unified health state. Finally, the stateful Digital Twin forecasts short-term risk and provides actionable recommendations before a glucose spike occurs."*

---

### Slide 4: From Health Data to a Digital Twin
* **Visual**: Split schematic showing Static Clinical Factors (left) and Dynamic Wearables (right) converging into the Personalized Health State.
* **Content**:
  * **Static Context**: Age, Gender, Weight / BMI, Medications, Baseline Sleep / Diagnosis.
  * **Dynamic Telemetry**: CGM glucose history, recent meals, sleep quality, pedometer activity steps.
  * *New data continually updates the representation so insights reflect the person's current context, not just population averages.*
* **Speaking Script**:
  > *"What turns data into a true Digital Twin? It is the continuous fusion of static phenotype and dynamic telemetry. Static parameters—such as the patient's age, BMI, baseline HbA1c, and current medications—anchor the twin's baseline metabolic sensitivity. Meanwhile, dynamic streams—such as 15-minute CGM readings, heart rate surges, and steps—continuously update the twin's live state in real time."*

---

### Slide 5: System Workflow
* **Visual**: Horizontal chevron flow highlighting the 7-second round-trip time.
* **Content**:
  * `Mobile App` $\longrightarrow$ `Backend APIs` $\longrightarrow$ `Prediction Engine` $\longrightarrow$ `User Dashboard`
  * **Key Flow**: Data collected on device $\rightarrow$ synced to backend $\rightarrow$ processed by scoring logic $\rightarrow$ Twin state updated $\rightarrow$ user/doctor sees contextual insight.
* **Speaking Script**:
  > *"Our end-to-end pipeline operates in near real-time: as sensor packets arrive from the user's wearable, they are ingested via FastAPI REST endpoints, validated against physiological laws, processed by our predictive feature engine, and reflected on the doctor's dashboard in under 7 seconds."*

---

### Slide 6: Technical Architecture
* **Visual**: System component cards and technology badges.
* **Content**:
  * **Frontend**: React Native • Expo • TypeScript / JavaScript (Mobile) + Streamlit (Doctor Dashboard).
  * **Backend**: Node.js / Python FastAPI • REST APIs.
  * **Database**: MongoDB / SQLite (user profiles, sensor timelines, prediction logs).
  * **AI & Tools**: Google Gemini (text/image multimodal nutrition assistance) + Scikit-learn / XGBoost (Predictive modeling).
  * **Integrations**: Apple Health / HealthKit, Barcode scanning, Open Food Facts nutrition database.
  * **Prediction**: GlycoTwin stateful prediction & scoring engine (TreeSHAP feature attributions).
* **Speaking Script**:
  > *"Under the hood, GlycoTwin is built with an industry-standard, decoupled architecture. Telemetry is ingested through REST APIs into our database. We leverage Open Food Facts and Gemini for automated meal recognition, while our machine learning and scoring engine computes explainable spike risks using TreeSHAP."*

---

### Slide 7: How We Estimate Glucose Impact
* **Visual**: 3-step numbered pipeline (01 Inputs $\rightarrow$ 02 Processing $\rightarrow$ 03 Output).
* **Content**:
  * **01 Inputs**: Meal (carbs, fiber, protein), recent glucose trend, activity steps, sleep quality, clinical profile.
  * **02 Processing**: Feature extraction $\rightarrow$ temporal kinematics ($\Delta G_{15/30}$, velocity) $\rightarrow$ trained model & scoring function.
  * **03 Output**: Glucose impact / 2-hour spike risk score used for clinical insight generation.
  * *Repository Note: Labeled as an interpretable proof-of-concept risk score. Clear educational disclaimer that synthetic/heuristic benchmarks are reported.*
* **Speaking Script**:
  > *"When estimating glycemic impact, our engine extracts retrospective temporal features: current glucose, velocity over the last 15 and 30 minutes, glycemic variability (%CV), and recent step counts. By feeding these into our trained model, we produce an interpretable 2-hour spike risk score alongside expected trajectory bounds."*

---

### Slide 8: From a Meal Photo to Personalized Insight
* **Visual**: Hands holding a smartphone scanning a balanced meal plate.
* **Content**:
  1. User captures meal photo or scans barcode.
  2. System queries barcode $\rightarrow$ Open Food Facts / Nutrition API lookup.
  3. If no barcode, image-assisted identification using Gemini or visual metadata.
  4. Nutrition values combined with user's recent physiological context (glucose trend, activity, sleep).
  5. GlycoTwin scoring engine estimates glycemic excursion impact.
  6. App surfaces immediate contextual suggestions (e.g., recommend a light 15-minute walk).
* **Speaking Script**:
  > *"Here is how a patient experiences GlycoTwin during a meal: they take a photo or scan a barcode. The system retrieves nutritional macronutrients, merges that with the patient's current glucose trend and prior night's sleep, and projects the excursion. If a spike is imminent, the twin suggests a targeted behavioral counter-action before hyperglycemia sets in."*

---

### Slide 9: Meet the Twin
* **Visual**: Prototype UI screenshots across mobile, tablet, and doctor dashboards.
* **Content**:
  * Screenshots showing real-time CGM curves, meal logging history, personalized daily summary, and clinical risk badges.
  * The Twin uses the user's own history to contextualize insights rather than only population rules.
* **Speaking Script**:
  > *"This slide showcases our working prototype UI. Clinicians and patients can inspect historical glucose trajectories, observe the 2-hour projected confidence envelope, review heart-rate and step activity bars, and view top risk drivers."*

---

### Slide 10: Why Personalization Matters
* **Visual**: Side-by-side contrast of Person A vs. Person B receiving the exact same meal.
* **Content**:
  * **Person A**: Good sleep • Higher recent activity • Glucose trending stable $\longrightarrow$ **Low Predicted Impact / Stable**.
  * **Person B**: Poor sleep • Low recent activity • Elevated baseline readings $\longrightarrow$ **High Predicted Spike Risk**.
  * **Summary**: *Same meal, different context $\rightarrow$ different predicted impact and tailored insight. Our focus is contextual interpretation, not definitive medical advice.*
* **Speaking Script**:
  > *"To summarize why a Digital Twin is essential: consider Person A and Person B eating the exact same plate of pasta. Person A slept 8 hours and took a walk—their glucose stays stable. Person B slept 4.5 hours and sat at a desk—their glucose spikes past 180 mg/dL. GlycoTwin understands this critical difference and provides the personalized context needed for proactive metabolic health."*

---

## 3. Video Presentation Checklist (20-Minute Video)

1. **Minutes 00:00 – 05:00**: Walk through Slides 1 through 4 (Motivation, Team, and Digital Twin Concept).
2. **Minutes 05:00 – 09:00**: Walk through Slides 5 through 8 (System Workflow, Architecture, and Meal-to-Insight pipeline).
3. **Minutes 09:00 – 16:00**: **Live Demo on Doctor Dashboard (`frontend/dashboard.py`)**:
   - Show virtual patient Arthur Pendelton (`PT-SYNTH-001`).
   - Trigger **Scenario 1 (Postprandial Surge)**: watch glucose climb ($118 \rightarrow 139 \rightarrow 168$ mg/dL) and risk jump to **High (78%)**.
   - Show the **TreeSHAP Risk Factor Breakdown** (+28% velocity, +22% carbs, +14% poor sleep).
   - Trigger **Scenario 2 (Euglycemic Walk)**: show how 1,420 steps blunt the spike and keep risk Low.
4. **Minutes 16:00 – 18:30**: Review Slides 9 and 10 (Meet the Twin & Why Personalization Matters).
5. **Minutes 18:30 – 20:00**: Conclude with Disclaimers, Limitations, and Q&A wrap-up.
