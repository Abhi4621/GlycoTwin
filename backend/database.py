"""
SQLite Database Layer for Type 2 Diabetes Digital Twin.
Stores patient clinical profiles, telemetry timelines, and prediction logs.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd


DB_PATH = "digital_twin.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes database tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Patients table (EHR profiles)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            bmi REAL NOT NULL,
            diagnosis TEXT NOT NULL,
            diabetes_duration_years REAL NOT NULL,
            hba1c REAL NOT NULL,
            fasting_glucose REAL NOT NULL,
            systolic_bp INTEGER NOT NULL,
            diastolic_bp INTEGER NOT NULL,
            medications_display TEXT,
            comorbidities TEXT,
            family_history INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Telemetry readings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            glucose REAL NOT NULL,
            heart_rate REAL NOT NULL,
            steps INTEGER NOT NULL,
            sleep_hours REAL NOT NULL,
            sleep_quality REAL NOT NULL,
            carbs_intake REAL DEFAULT 0.0,
            activity_type TEXT DEFAULT 'sedentary',
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
    """)

    # Twin prediction logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS twin_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            spike_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            predicted_trajectory TEXT,
            risk_factors TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
    """)

    conn.commit()
    conn.close()


def save_patient(patient_dict: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """Upserts a patient EHR profile."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO patients (
            patient_id, name, age, gender, bmi, diagnosis, diabetes_duration_years,
            hba1c, fasting_glucose, systolic_bp, diastolic_bp, medications_display,
            comorbidities, family_history
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_dict.get("patient_id"),
        patient_dict.get("name", "Arthur Pendelton"),
        int(patient_dict.get("age", 58)),
        patient_dict.get("gender", "Male"),
        float(patient_dict.get("bmi", 29.5)),
        patient_dict.get("diagnosis", "Type 2 Diabetes Mellitus"),
        float(patient_dict.get("diabetes_duration_years", 6.0)),
        float(patient_dict.get("hba1c", 8.1)),
        float(patient_dict.get("fasting_glucose", 138.0)),
        int(patient_dict.get("systolic_bp", 134)),
        int(patient_dict.get("diastolic_bp", 86)),
        patient_dict.get("medications_display", "Metformin 1000mg BID"),
        patient_dict.get("comorbidities", "Hypertension"),
        int(patient_dict.get("family_history_diabetes", 1)),
    ))
    conn.commit()
    conn.close()


def get_patient(patient_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves patient EHR record by ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def insert_sensor_reading(reading: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """Appends a validated sensor reading to the database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sensor_readings (
            patient_id, timestamp, glucose, heart_rate, steps, sleep_hours, sleep_quality, carbs_intake, activity_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reading.get("patient_id", "PT-SYNTH-001"),
        str(reading.get("timestamp")),
        float(reading.get("glucose")),
        float(reading.get("heart_rate", 72.0)),
        int(reading.get("steps", 0)),
        float(reading.get("sleep_hours", 7.0)),
        float(reading.get("sleep_quality", 70.0)),
        float(reading.get("carbs_intake", 0.0)),
        str(reading.get("activity_type", "sedentary")),
    ))
    conn.commit()
    conn.close()


def get_patient_history(patient_id: str, limit: int = 96, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent sensor history for a patient in chronological order."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM (
            SELECT * FROM sensor_readings WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?
        ) ORDER BY timestamp ASC
    """, (patient_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_prediction(
    patient_id: str,
    timestamp: str,
    spike_prob: float,
    risk_level: str,
    trajectory: List[float],
    risk_factors: List[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> None:
    """Logs model prediction and risk factors."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO twin_predictions (
            patient_id, timestamp, spike_probability, risk_level, predicted_trajectory, risk_factors
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        timestamp,
        float(spike_prob),
        risk_level,
        json.dumps(trajectory),
        json.dumps(risk_factors),
    ))
    conn.commit()
    conn.close()
