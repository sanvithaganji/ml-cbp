"""
IoT Telemetry Ingestion Receiver.
Receives HTTP POST requests from ESP32 microclimate nodes,
validates readings against physical sensor bounds, and logs to SQLite and JSON.
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.validation import DEFAULT_BOUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "outputs/sensor_telemetry.db"
LATEST_JSON_PATH = "outputs/latest_telemetry.json"


def init_telemetry_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            device_id TEXT,
            hardware_mode TEXT,
            ambient_temperature REAL,
            soil_temperature REAL,
            humidity REAL,
            soil_moisture REAL,
            leaf_wetness REAL,
            is_valid INTEGER,
            validation_notes TEXT
        )
    """)
    conn.commit()
    conn.close()


def store_telemetry_reading(payload: Dict[str, Any], db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Validate, store in SQLite, and update latest_telemetry.json.
    """
    init_telemetry_db(db_path)
    
    device_id = payload.get("device_id", "UNKNOWN_ESP32")
    hw_mode = payload.get("hardware_mode", "PHYSICAL_ESP32")
    telemetry = payload.get("telemetry", {})
    ts = datetime.utcnow().isoformat() + "Z"

    validation_notes = []
    is_valid = True

    cleaned = {}
    for feat, (b_min, b_max) in DEFAULT_BOUNDS.items():
        val = telemetry.get(feat)
        if val is None:
            cleaned[feat] = None
            validation_notes.append(f"{feat}: DISCONNECTED/NULL")
        else:
            try:
                f_val = float(val)
                if b_min <= f_val <= b_max:
                    cleaned[feat] = round(f_val, 2)
                else:
                    cleaned[feat] = round(f_val, 2)
                    is_valid = False
                    validation_notes.append(f"{feat}: OUT_OF_BOUNDS ({f_val})")
            except (ValueError, TypeError):
                cleaned[feat] = None
                validation_notes.append(f"{feat}: INVALID_TYPE")

    # Save to SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO telemetry (
            timestamp, device_id, hardware_mode,
            ambient_temperature, soil_temperature, humidity, soil_moisture, leaf_wetness,
            is_valid, validation_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, device_id, hw_mode,
        cleaned["ambient_temperature"], cleaned["soil_temperature"],
        cleaned["humidity"], cleaned["soil_moisture"], cleaned["leaf_wetness"],
        1 if is_valid else 0, "; ".join(validation_notes)
    ))
    conn.commit()
    conn.close()

    # Update latest JSON
    record = {
        "timestamp": ts,
        "device_id": device_id,
        "hardware_mode": hw_mode,
        "telemetry": cleaned,
        "is_valid": is_valid,
        "validation_notes": validation_notes,
    }
    os.makedirs(os.path.dirname(LATEST_JSON_PATH), exist_ok=True)
    with open(LATEST_JSON_PATH, "w") as f:
        json.dump(record, f, indent=2)

    return record


def get_latest_telemetry(json_path: str = LATEST_JSON_PATH) -> Optional[Dict[str, Any]]:
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_telemetry_history(db_path: str = DB_PATH, limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


class TelemetryHTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/telemetry":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                record = store_telemetry_reading(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {"status": "success", "received": record}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                logger.info("Ingested telemetry from %s: %s", record["device_id"], record["telemetry"])
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                err = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(err).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/api/telemetry/latest":
            latest = get_latest_telemetry()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(latest or {}).encode("utf-8"))
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ESP32 Telemetry Receiver Active.")


def run_receiver(port: int = 8080):
    init_telemetry_db()
    server_address = ("", port)
    httpd = HTTPServer(server_address, TelemetryHTTPHandler)
    logger.info("Starting ESP32 Telemetry HTTP Server on port %d...", port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down telemetry receiver.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()
    run_receiver(args.port)
