"""
Database module for the Cloud-Based Driver Behavior Analytics Platform.
Handles local SQLite storage and queries for telemetry, events, and scores.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "driver_analytics.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes SQLite tables if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Telemetry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                trip_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                speed REAL NOT NULL,
                acceleration REAL NOT NULL,
                braking_intensity REAL NOT NULL,
                steering_angle REAL NOT NULL,
                distance_travelled REAL NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_driver ON telemetry(driver_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_trip ON telemetry(trip_id);")

        # Driving Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS driving_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                driver_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                trip_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                value REAL NOT NULL,
                description TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_driver ON driving_events(driver_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_trip ON driving_events(trip_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON driving_events(event_type);")

        # Driver Scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS driver_scores (
                driver_id TEXT PRIMARY KEY,
                safety_score REAL NOT NULL,
                risk_category TEXT NOT NULL,
                total_trips INTEGER NOT NULL,
                total_distance_km REAL NOT NULL,
                total_events INTEGER NOT NULL,
                overspeeding_count INTEGER NOT NULL,
                harsh_braking_count INTEGER NOT NULL,
                rapid_accel_count INTEGER NOT NULL,
                sharp_turning_count INTEGER NOT NULL,
                excessive_idling_count INTEGER NOT NULL,
                avg_speed REAL NOT NULL,
                max_speed REAL NOT NULL,
                last_calculated TEXT NOT NULL
            );
        """)

        # Model evaluation metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                samples_count INTEGER NOT NULL,
                notes TEXT
            );
        """)
        conn.commit()


def save_telemetry(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH, if_exists: str = "append") -> int:
    """Saves telemetry dataframe to database."""
    if df.empty:
        return 0
    with get_connection(db_path) as conn:
        df.to_sql("telemetry", conn, if_exists=if_exists, index=False)
        return len(df)


def save_events(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH, if_exists: str = "append") -> int:
    """Saves detected events to database."""
    if df.empty:
        return 0
    with get_connection(db_path) as conn:
        df.to_sql("driving_events", conn, if_exists=if_exists, index=False)
        return len(df)


def save_driver_scores(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """Saves or updates driver safety scores in database."""
    if df.empty:
        return 0
    with get_connection(db_path) as conn:
        # Replace driver_scores table or upsert
        df.to_sql("driver_scores", conn, if_exists="replace", index=False)
        return len(df)


def save_model_metric(metrics: dict, db_path: str = DEFAULT_DB_PATH) -> None:
    """Saves model training metrics to database."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_metrics 
            (model_name, trained_at, accuracy, precision, recall, f1_score, samples_count, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.get("model_name", "Random Forest Classifier"),
            metrics.get("trained_at", datetime.now().isoformat()),
            metrics.get("accuracy", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1_score", 0.0),
            metrics.get("samples_count", 0),
            metrics.get("notes", "")
        ))
        conn.commit()


def get_telemetry(trip_id: str = None, driver_id: str = None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves telemetry data with optional filtering."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM telemetry WHERE 1=1"
        params = []
        if trip_id:
            query += " AND trip_id = ?"
            params.append(trip_id)
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        query += " ORDER BY timestamp ASC"
        return pd.read_sql_query(query, conn, params=params)


def get_events(driver_id: str = None, trip_id: str = None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves detected driving events with optional filtering."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM driving_events WHERE 1=1"
        params = []
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        if trip_id:
            query += " AND trip_id = ?"
            params.append(trip_id)
        query += " ORDER BY timestamp ASC"
        return pd.read_sql_query(query, conn, params=params)


def get_driver_scores(driver_id: str = None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves computed driver safety scores."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM driver_scores WHERE 1=1"
        params = []
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        query += " ORDER BY safety_score DESC"
        return pd.read_sql_query(query, conn, params=params)


def get_latest_model_metrics(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Retrieves the most recent model evaluation metrics."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM model_metrics ORDER BY id DESC LIMIT 1"
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}


def get_fleet_summary(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Calculates high-level fleet KPIs."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(DISTINCT driver_id), COUNT(DISTINCT trip_id) FROM telemetry")
        row = cursor.fetchone()
        total_drivers = row[0] if row and row[0] is not None else 0
        total_trips = row[1] if row and row[1] is not None else 0

        cursor.execute("SELECT AVG(safety_score) FROM driver_scores")
        avg_row = cursor.fetchone()
        avg_score = round(avg_row[0], 1) if avg_row and avg_row[0] is not None else 0.0

        cursor.execute("SELECT COUNT(*) FROM driving_events")
        ev_row = cursor.fetchone()
        total_events = ev_row[0] if ev_row and ev_row[0] is not None else 0

        return {
            "total_drivers": total_drivers,
            "total_trips": total_trips,
            "avg_safety_score": avg_score,
            "total_unsafe_events": total_events
        }


def clear_all_data(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clears all table rows in the database."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM telemetry;")
        cursor.execute("DELETE FROM driving_events;")
        cursor.execute("DELETE FROM driver_scores;")
        cursor.execute("DELETE FROM model_metrics;")
        conn.commit()
