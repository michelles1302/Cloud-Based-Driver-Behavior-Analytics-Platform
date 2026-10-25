"""
Pipeline orchestration module for Driver Behavior Analytics Platform.
Coordinates telemetry generation, unsafe behavior detection, safety scoring,
and SQLite database synchronization in a unified workflow.
"""

import pandas as pd
from typing import Dict, Any, Optional

import database
from telemetry_generator import generate_fleet_telemetry, save_telemetry_to_csv
from behavior_detector import detect_all_events, DEFAULT_THRESHOLDS
from safety_score import calculate_driver_safety_scores


def run_full_pipeline(
    num_drivers: int = 5,
    trips_per_driver: int = 2,
    trip_duration_seconds: int = 150,
    custom_thresholds: Optional[Dict[str, Any]] = None,
    reset_db: bool = True
) -> Dict[str, Any]:
    """
    Executes the end-to-end telemetry analytics pipeline:
    1. Initializes database schema
    2. Generates realistic multi-driver driving timeseries
    3. Detects unsafe events (overspeeding, harsh braking, rapid accel, sharp turns, excessive idling)
    4. Computes 0-100 driver safety scores and risk categories
    5. Persists all datasets to local SQLite storage and CSV
    """
    database.init_db()

    if reset_db:
        database.clear_all_data()

    # 1. Telemetry Simulation
    telemetry_df = generate_fleet_telemetry(
        num_drivers=num_drivers,
        trips_per_driver=trips_per_driver,
        trip_duration_seconds=trip_duration_seconds
    )
    save_telemetry_to_csv(telemetry_df)
    database.save_telemetry(telemetry_df)

    # 2. Behavior Detection
    thresholds = custom_thresholds or DEFAULT_THRESHOLDS
    events_df = detect_all_events(telemetry_df, thresholds=thresholds)
    database.save_events(events_df)

    # 3. Safety Scoring
    scores_df = calculate_driver_safety_scores(telemetry_df, events_df)
    database.save_driver_scores(scores_df)

    # 4. Fleet Summary
    fleet_summary = database.get_fleet_summary()

    return {
        "status": "success",
        "telemetry_count": len(telemetry_df),
        "events_count": len(events_df),
        "drivers_count": len(scores_df),
        "trips_count": telemetry_df["trip_id"].nunique(),
        "avg_safety_score": fleet_summary["avg_safety_score"],
        "scores_df": scores_df,
        "events_df": events_df,
        "telemetry_df": telemetry_df
    }


if __name__ == "__main__":
    print("Running Driver Analytics Pipeline...")
    res = run_full_pipeline(num_drivers=5, trips_per_driver=2, trip_duration_seconds=150)
    print("\n--- Pipeline Execution Summary ---")
    print(f"Total Drivers:     {res['drivers_count']}")
    print(f"Total Trips:       {res['trips_count']}")
    print(f"Telemetry Points:  {res['telemetry_count']}")
    print(f"Detected Events:   {res['events_count']}")
    print(f"Fleet Avg Score:   {res['avg_safety_score']}")
    print("\nDriver Scores Summary:")
    print(res["scores_df"][["driver_id", "safety_score", "risk_category", "total_events", "avg_speed", "max_speed"]])
