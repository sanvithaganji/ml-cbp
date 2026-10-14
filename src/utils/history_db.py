"""
Prediction History SQLite Database Utility.
Logs user diagnostic predictions locally with timestamps, model selection,
confidence scores, sensor parameters, and export to CSV.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

DEFAULT_DB = "outputs/predictions_history.db"


def init_prediction_db(db_path: str = DEFAULT_DB):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            has_sensor_data INTEGER NOT NULL,
            ambient_temperature REAL,
            soil_temperature REAL,
            humidity REAL,
            soil_moisture REAL,
            leaf_wetness REAL,
            prob_healthy REAL,
            prob_anthracnose REAL,
            prob_leaf_spot REAL,
            composite_risk_score REAL,
            risk_category TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(result: Dict[str, Any], sensor_dict: Optional[Dict[str, float]], db_path: str = DEFAULT_DB):
    init_prediction_db(db_path)
    ts = datetime.utcnow().isoformat() + "Z"

    probs = result.get("probabilities", {})
    risk = result.get("risk_assessment") or {}
    s = sensor_dict or {}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (
            timestamp, model_used, predicted_class, confidence, has_sensor_data,
            ambient_temperature, soil_temperature, humidity, soil_moisture, leaf_wetness,
            prob_healthy, prob_anthracnose, prob_leaf_spot,
            composite_risk_score, risk_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts,
        result.get("model_used", "Unknown"),
        result.get("predicted_class", "Unknown"),
        float(result.get("confidence", 0.0)),
        1 if result.get("has_sensor_data") else 0,
        s.get("ambient_temperature"),
        s.get("soil_temperature"),
        s.get("humidity"),
        s.get("soil_moisture"),
        s.get("leaf_wetness"),
        float(probs.get("Healthy", 0.0)),
        float(probs.get("Anthracnose", 0.0)),
        float(probs.get("Leaf Spot", 0.0)),
        float(risk.get("composite_risk_score")) if risk.get("composite_risk_score") is not None else None,
        risk.get("risk_category"),
    ))
    conn.commit()
    conn.close()


def fetch_prediction_history(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    init_prediction_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df
