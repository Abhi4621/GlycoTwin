"""
Synthetic EHR Cohort Generator for Type 2 Diabetes Patients.
Follows realistic clinical distributions modeled after Synthea & NHANES demographics.
All generated records are purely synthetic for research and education.
"""

import argparse
import os
from typing import Optional
import numpy as np
import pandas as pd


FIRST_NAMES_MALE = [
    "Arthur", "Marcus", "David", "Robert", "James", "William", "Charles", "Thomas",
    "Richard", "Joseph", "Daniel", "Michael", "Alexander", "Suresh", "Ramesh", "Wei",
    "Carlos", "Ahmed", "Diego", "Liam"
]
FIRST_NAMES_FEMALE = [
    "Eleanor", "Margaret", "Patricia", "Linda", "Barbara", "Elizabeth", "Jennifer",
    "Susan", "Dorothy", "Sarah", "Priya", "Sunita", "Mei", "Elena", "Fatima",
    "Maria", "Grace", "Chloe", "Emma", "Amina"
]
LAST_NAMES = [
    "Pendelton", "Vance", "Brody", "Sterling", "Holloway", "Chen", "Patel", "Sharma",
    "Rodriguez", "Hernandez", "Kim", "Tanaka", "Al-Mansoor", "Dubois", "Schmidt",
    "O'Connor", "Kowalski", "Johnson", "Smith", "Williams"
]


def generate_ehr_cohort(num_patients: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic cohort of Type 2 Diabetes patients with correlated clinical features.

    Parameters:
        num_patients: Number of synthetic patient profiles to generate.
        seed: Random seed for exact reproducibility.

    Returns:
        pd.DataFrame containing synthetic patient profiles.
    """
    rng = np.random.default_rng(seed)

    records = []
    for i in range(1, num_patients + 1):
        patient_id = f"PT-SYNTH-{i:03d}"
        gender = rng.choice(["Male", "Female"], p=[0.52, 0.48])

        if gender == "Male":
            first_name = rng.choice(FIRST_NAMES_MALE)
        else:
            first_name = rng.choice(FIRST_NAMES_FEMALE)
        last_name = rng.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"

        # Age distribution: T2D adult cohort 38 to 78, mean ~58
        age = int(np.clip(rng.normal(loc=58.5, scale=9.5), 35, 82))

        # BMI distribution: overweight/obese prevalence in T2D
        bmi = float(np.round(np.clip(rng.normal(loc=29.8, scale=4.8), 21.0, 44.0), 1))

        # Duration of diabetes
        diabetes_duration = float(np.round(np.clip(rng.exponential(scale=5.0) + 1.0, 1.0, 26.0), 1))

        # HbA1c correlated with diabetes duration and BMI
        hba1c_latent = 6.2 + 0.04 * (bmi - 25.0) + 0.05 * diabetes_duration + rng.normal(0, 0.6)
        hba1c = float(np.round(np.clip(hba1c_latent, 6.2, 11.5), 1))

        # Fasting glucose correlated with HbA1c
        fasting_glucose = int(np.round(np.clip(28.7 * hba1c - 80.0 + rng.normal(0, 12), 100, 250)))

        # Blood pressure (Hypertension frequently comorbid with T2D)
        systolic_bp = int(np.clip(rng.normal(loc=132 + (age - 50) * 0.4, scale=12), 110, 175))
        diastolic_bp = int(np.clip(rng.normal(loc=82 + (age - 50) * 0.15, scale=8), 65, 105))

        # Medications: standard escalation pathway
        has_metformin = int(rng.uniform() < 0.88)
        has_sglt2i = int(rng.uniform() < 0.42 if hba1c > 7.5 else 0.20)
        has_glp1 = int(rng.uniform() < 0.35 if bmi > 30.0 else 0.15)
        has_sulfonylurea = int(rng.uniform() < 0.28 if diabetes_duration > 6 else 0.10)
        has_insulin = int(rng.uniform() < 0.30 if (hba1c > 8.5 or diabetes_duration > 10) else 0.05)

        med_list = []
        if has_metformin:
            med_list.append("Metformin 1000mg BID")
        if has_sglt2i:
            med_list.append("Empagliflozin 10mg QD")
        if has_glp1:
            med_list.append("Semaglutide 0.5mg QW")
        if has_sulfonylurea:
            med_list.append("Glimepiride 2mg QD")
        if has_insulin:
            med_list.append("Glargine U-100 20u QHS")
        if not med_list:
            med_list.append("Metformin 500mg BID")
            has_metformin = 1

        medications_display = ", ".join(med_list)

        # Comorbidities
        comorbidities_list = []
        if systolic_bp >= 130 or diastolic_bp >= 80:
            comorbidities_list.append("Essential Hypertension")
        if bmi >= 30.0 or rng.uniform() < 0.65:
            comorbidities_list.append("Dyslipidemia")
        if diabetes_duration > 8 and rng.uniform() < 0.30:
            comorbidities_list.append("Mild Non-Proliferative Retinopathy")
        comorbidities = ", ".join(comorbidities_list) if comorbidities_list else "None recorded"

        family_history = int(rng.uniform() < 0.68)
        smoking = rng.choice(["Never", "Former", "Current"], p=[0.55, 0.32, 0.13])

        records.append({
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "diagnosis": "Type 2 Diabetes Mellitus",
            "diabetes_duration_years": diabetes_duration,
            "hba1c": hba1c,
            "fasting_glucose": fasting_glucose,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "medication_metformin": has_metformin,
            "medication_sglt2i": has_sglt2i,
            "medication_glp1": has_glp1,
            "medication_sulfonylurea": has_sulfonylurea,
            "medication_insulin": has_insulin,
            "medications_display": medications_display,
            "comorbidities": comorbidities,
            "family_history_diabetes": family_history,
            "smoking_status": smoking,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic EHR cohort for T2D Digital Twin.")
    parser.add_argument("--num_patients", type=int, default=100, help="Number of patients to generate")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_ehr_patients.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_ehr = generate_ehr_cohort(num_patients=args.num_patients, seed=args.seed)
    df_ehr.to_csv(args.output, index=False)
    print(f"Successfully generated {len(df_ehr)} synthetic EHR profiles -> {args.output}")
