"""
Behavior Detector module for Driver Behavior Analytics Platform.
Applies configurable rule-based thresholds to detect unsafe driving patterns:
- Overspeeding
- Harsh braking
- Rapid acceleration
- Sharp turning
- Excessive idling
"""

import uuid
import pandas as pd
from typing import Dict, List, Any


# Default threshold parameters (easily configurable)
DEFAULT_THRESHOLDS = {
    "speed_limit_kmh": 100.0,
    "harsh_brake_accel_ms2": -3.0,
    "rapid_accel_ms2": 3.0,
    "sharp_turn_angle_deg": 28.0,
    "sharp_turn_min_speed_kmh": 25.0,
    "excessive_idle_seconds": 60
}


def classify_severity(event_type: str, value: float) -> str:
    """Classifies event severity into Low, Medium, or High based on value magnitude."""
    abs_val = abs(value)
    if event_type == "overspeeding":
        if abs_val < 112.0:
            return "Low"
        elif abs_val < 125.0:
            return "Medium"
        return "High"

    elif event_type == "harsh_braking":
        # Acceleration is negative, magnitude is abs_val
        if abs_val < 3.6:
            return "Low"
        elif abs_val < 4.4:
            return "Medium"
        return "High"

    elif event_type == "rapid_acceleration":
        if abs_val < 3.6:
            return "Low"
        elif abs_val < 4.3:
            return "Medium"
        return "High"

    elif event_type == "sharp_turning":
        if abs_val < 33.0:
            return "Low"
        elif abs_val < 39.0:
            return "Medium"
        return "High"

    elif event_type == "excessive_idling":
        if abs_val < 75.0:
            return "Low"
        elif abs_val < 90.0:
            return "Medium"
        return "High"

    return "Low"


def detect_trip_events(trip_df: pd.DataFrame, thresholds: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Analyzes telemetry for a single trip and extracts unsafe driving events.
    Groups contiguous event blocks (like ongoing overspeeding or prolonged idling).
    """
    cfg = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        cfg.update(thresholds)

    events = []
    if trip_df.empty:
        return events

    # Sort sequentially by timestamp or index
    sorted_df = trip_df.sort_values(by="timestamp").reset_index(drop=True)
    driver_id = sorted_df["driver_id"].iloc[0]
    vehicle_id = sorted_df["vehicle_id"].iloc[0]
    trip_id = sorted_df["trip_id"].iloc[0]

    n_rows = len(sorted_df)
    i = 0

    # Tracking for contiguous event clusters
    in_brake = False
    brake_start_idx = -1
    min_brake_accel = 0.0

    in_accel = False
    accel_start_idx = -1
    max_accel_val = 0.0

    in_turn = False
    turn_start_idx = -1
    max_turn_angle = 0.0

    # Consecutive idle tracking
    idle_streak = 0
    idle_start_idx = -1

    # Consecutive overspeeding tracking
    speed_streak = 0
    speed_start_idx = -1
    max_streak_speed = 0.0

    while i < n_rows:
        row = sorted_df.iloc[i]
        speed = float(row["speed"])
        accel = float(row["acceleration"])
        angle = float(row["steering_angle"])
        timestamp = str(row["timestamp"])

        # ---------------------------------------------------------
        # 1. Harsh Braking Check (Cluster consecutive ticks)
        # ---------------------------------------------------------
        if accel <= cfg["harsh_brake_accel_ms2"] and speed > 10.0:
            if not in_brake:
                in_brake = True
                brake_start_idx = i
                min_brake_accel = accel
            else:
                min_brake_accel = min(min_brake_accel, accel)
        else:
            if in_brake:
                sev = classify_severity("harsh_braking", min_brake_accel)
                start_ts = sorted_df.iloc[brake_start_idx]["timestamp"]
                events.append({
                    "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
                    "driver_id": driver_id,
                    "vehicle_id": vehicle_id,
                    "trip_id": trip_id,
                    "timestamp": start_ts,
                    "event_type": "harsh_braking",
                    "severity": sev,
                    "value": round(min_brake_accel, 2),
                    "description": f"Harsh braking maneuver: Peak deceleration {min_brake_accel:.2f} m/s²"
                })
                in_brake = False
                min_brake_accel = 0.0

        # ---------------------------------------------------------
        # 2. Rapid Acceleration Check (Cluster consecutive ticks)
        # ---------------------------------------------------------
        if accel >= cfg["rapid_accel_ms2"]:
            if not in_accel:
                in_accel = True
                accel_start_idx = i
                max_accel_val = accel
            else:
                max_accel_val = max(max_accel_val, accel)
        else:
            if in_accel:
                sev = classify_severity("rapid_acceleration", max_accel_val)
                start_ts = sorted_df.iloc[accel_start_idx]["timestamp"]
                events.append({
                    "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
                    "driver_id": driver_id,
                    "vehicle_id": vehicle_id,
                    "trip_id": trip_id,
                    "timestamp": start_ts,
                    "event_type": "rapid_acceleration",
                    "severity": sev,
                    "value": round(max_accel_val, 2),
                    "description": f"Rapid acceleration burst: Peak {max_accel_val:.2f} m/s²"
                })
                in_accel = False
                max_accel_val = 0.0

        # ---------------------------------------------------------
        # 3. Sharp Turn Check (Cluster consecutive ticks)
        # ---------------------------------------------------------
        if abs(angle) >= cfg["sharp_turn_angle_deg"] and speed >= cfg["sharp_turn_min_speed_kmh"]:
            if not in_turn:
                in_turn = True
                turn_start_idx = i
                max_turn_angle = angle
            else:
                if abs(angle) > abs(max_turn_angle):
                    max_turn_angle = angle
        else:
            if in_turn:
                sev = classify_severity("sharp_turning", max_turn_angle)
                start_ts = sorted_df.iloc[turn_start_idx]["timestamp"]
                events.append({
                    "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
                    "driver_id": driver_id,
                    "vehicle_id": vehicle_id,
                    "trip_id": trip_id,
                    "timestamp": start_ts,
                    "event_type": "sharp_turning",
                    "severity": sev,
                    "value": round(max_turn_angle, 1),
                    "description": f"Sharp steering maneuver: Peak {max_turn_angle:.1f}° at {speed:.1f} km/h"
                })
                in_turn = False
                max_turn_angle = 0.0

        # ---------------------------------------------------------
        # 4. Overspeeding (Grouped consecutive ticks >= 3s)
        # ---------------------------------------------------------
        if speed > cfg["speed_limit_kmh"]:
            if speed_streak == 0:
                speed_start_idx = i
                max_streak_speed = speed
            else:
                max_streak_speed = max(max_streak_speed, speed)
            speed_streak += 1
        else:
            if speed_streak >= 3:
                sev = classify_severity("overspeeding", max_streak_speed)
                start_ts = sorted_df.iloc[speed_start_idx]["timestamp"]
                events.append({
                    "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
                    "driver_id": driver_id,
                    "vehicle_id": vehicle_id,
                    "trip_id": trip_id,
                    "timestamp": start_ts,
                    "event_type": "overspeeding",
                    "severity": sev,
                    "value": round(max_streak_speed, 1),
                    "description": f"Overspeeding episode: Peak {max_streak_speed:.1f} km/h for {speed_streak}s"
                })
            speed_streak = 0
            max_streak_speed = 0.0

        # ---------------------------------------------------------
        # 5. Excessive Idling (Consecutive stationary ticks)
        # ---------------------------------------------------------
        if speed == 0.0 and abs(accel) < 0.1:
            if idle_streak == 0:
                idle_start_idx = i
            idle_streak += 1
        else:
            if idle_streak >= cfg["excessive_idle_seconds"]:
                sev = classify_severity("excessive_idling", float(idle_streak))
                start_ts = sorted_df.iloc[idle_start_idx]["timestamp"]
                events.append({
                    "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
                    "driver_id": driver_id,
                    "vehicle_id": vehicle_id,
                    "trip_id": trip_id,
                    "timestamp": start_ts,
                    "event_type": "excessive_idling",
                    "severity": sev,
                    "value": float(idle_streak),
                    "description": f"Excessive stationary idling: Engine active without motion for {idle_streak}s"
                })
            idle_streak = 0

        i += 1

    # End of trip flush for ongoing streaks
    if in_brake:
        sev = classify_severity("harsh_braking", min_brake_accel)
        start_ts = sorted_df.iloc[brake_start_idx]["timestamp"]
        events.append({
            "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
            "driver_id": driver_id,
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "timestamp": start_ts,
            "event_type": "harsh_braking",
            "severity": sev,
            "value": round(min_brake_accel, 2),
            "description": f"Harsh braking maneuver: Peak deceleration {min_brake_accel:.2f} m/s²"
        })

    if in_accel:
        sev = classify_severity("rapid_acceleration", max_accel_val)
        start_ts = sorted_df.iloc[accel_start_idx]["timestamp"]
        events.append({
            "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
            "driver_id": driver_id,
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "timestamp": start_ts,
            "event_type": "rapid_acceleration",
            "severity": sev,
            "value": round(max_accel_val, 2),
            "description": f"Rapid acceleration burst: Peak {max_accel_val:.2f} m/s²"
        })

    if in_turn:
        sev = classify_severity("sharp_turning", max_turn_angle)
        start_ts = sorted_df.iloc[turn_start_idx]["timestamp"]
        events.append({
            "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
            "driver_id": driver_id,
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "timestamp": start_ts,
            "event_type": "sharp_turning",
            "severity": sev,
            "value": round(max_turn_angle, 1),
            "description": f"Sharp steering maneuver: Peak {max_turn_angle:.1f}°"
        })

    if speed_streak >= 3:
        sev = classify_severity("overspeeding", max_streak_speed)
        start_ts = sorted_df.iloc[speed_start_idx]["timestamp"]
        events.append({
            "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
            "driver_id": driver_id,
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "timestamp": start_ts,
            "event_type": "overspeeding",
            "severity": sev,
            "value": round(max_streak_speed, 1),
            "description": f"Overspeeding episode: Peak {max_streak_speed:.1f} km/h for {speed_streak}s"
        })

    if idle_streak >= cfg["excessive_idle_seconds"]:
        sev = classify_severity("excessive_idling", float(idle_streak))
        start_ts = sorted_df.iloc[idle_start_idx]["timestamp"]
        events.append({
            "event_id": f"EVT-{trip_id}-{len(events)+1:02d}",
            "driver_id": driver_id,
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "timestamp": start_ts,
            "event_type": "excessive_idling",
            "severity": sev,
            "value": float(idle_streak),
            "description": f"Excessive stationary idling: {idle_streak}s idle"
        })

    return events


def detect_all_events(telemetry_df: pd.DataFrame, thresholds: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Processes the entire fleet telemetry dataframe and returns detected events dataframe.
    """
    if telemetry_df.empty:
        return pd.DataFrame(columns=[
            "event_id", "driver_id", "vehicle_id", "trip_id",
            "timestamp", "event_type", "severity", "value", "description"
        ])

    all_events = []
    # Process trip by trip
    for trip_id, trip_group in telemetry_df.groupby("trip_id"):
        trip_events = detect_trip_events(trip_group, thresholds=thresholds)
        all_events.extend(trip_events)

    if not all_events:
        return pd.DataFrame(columns=[
            "event_id", "driver_id", "vehicle_id", "trip_id",
            "timestamp", "event_type", "severity", "value", "description"
        ])

    events_df = pd.DataFrame(all_events)
    return events_df
