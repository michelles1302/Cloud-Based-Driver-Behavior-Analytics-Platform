"""
Safety Score module for Driver Behavior Analytics Platform.
Calculates transparent 0-100 safety score and classifies risk categories:
- 90-100: Excellent
- 75-89: Good
- 60-74: Moderate
- Below 60: High Risk
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Event penalty deduction weights (per event, adjusted by severity multiplier)
BASE_PENALTIES = {
    "overspeeding": 4.5,
    "harsh_braking": 5.0,
    "rapid_acceleration": 3.0,
    "sharp_turning": 3.5,
    "excessive_idling": 2.5
}

SEVERITY_MULTIPLIER = {
    "Low": 0.8,
    "Medium": 1.0,
    "High": 1.3
}


def classify_risk_category(score: float) -> str:
    """Classifies numerical safety score into risk band."""
    if score >= 90.0:
        return "Excellent"
    elif score >= 75.0:
        return "Good"
    elif score >= 60.0:
        return "Moderate"
    else:
        return "High Risk"


def calculate_driver_safety_scores(
    telemetry_df: pd.DataFrame,
    events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes overall safety score (0 to 100) and aggregates driving metrics
    for each driver in the fleet.
    """
    if telemetry_df.empty:
        return pd.DataFrame(columns=[
            "driver_id", "safety_score", "risk_category", "total_trips",
            "total_distance_km", "total_events", "overspeeding_count",
            "harsh_braking_count", "rapid_accel_count", "sharp_turning_count",
            "excessive_idling_count", "avg_speed", "max_speed", "last_calculated"
        ])

    driver_rows = []
    drivers = telemetry_df["driver_id"].unique()

    for d_id in sorted(drivers):
        d_telemetry = telemetry_df[telemetry_df["driver_id"] == d_id]
        d_events = events_df[events_df["driver_id"] == d_id] if not events_df.empty else pd.DataFrame()

        # Fleet / Driver trip stats
        trips = d_telemetry["trip_id"].nunique()
        # Sum distance per trip (max distance reached in each trip)
        total_distance = round(d_telemetry.groupby("trip_id")["distance_travelled"].max().sum(), 2)
        avg_speed = round(float(d_telemetry["speed"].mean()), 1)
        max_speed = round(float(d_telemetry["speed"].max()), 1)

        # Event counts by category
        if not d_events.empty:
            total_events = len(d_events)
            overspeed_cnt = len(d_events[d_events["event_type"] == "overspeeding"])
            harsh_brake_cnt = len(d_events[d_events["event_type"] == "harsh_braking"])
            rapid_accel_cnt = len(d_events[d_events["event_type"] == "rapid_acceleration"])
            sharp_turn_cnt = len(d_events[d_events["event_type"] == "sharp_turning"])
            idle_cnt = len(d_events[d_events["event_type"] == "excessive_idling"])

            # Compute penalty deductions with severity weighting
            total_penalty = 0.0
            for _, ev in d_events.iterrows():
                base_p = BASE_PENALTIES.get(ev["event_type"], 3.0)
                sev_m = SEVERITY_MULTIPLIER.get(ev.get("severity", "Medium"), 1.0)
                total_penalty += base_p * sev_m
        else:
            total_events = 0
            overspeed_cnt = 0
            harsh_brake_cnt = 0
            rapid_accel_cnt = 0
            sharp_turn_cnt = 0
            idle_cnt = 0
            total_penalty = 0.0

        # Calculate score clamped between 0 and 100
        # Normalize penalties per trip so metrics scale fairly with trip volume
        trip_normalization = max(1.0, float(trips))
        effective_penalty = total_penalty / trip_normalization

        raw_score = 100.0 - effective_penalty
        final_score = round(max(0.0, min(100.0, raw_score)), 1)
        risk_category = classify_risk_category(final_score)

        driver_rows.append({
            "driver_id": d_id,
            "safety_score": final_score,
            "risk_category": risk_category,
            "total_trips": int(trips),
            "total_distance_km": float(total_distance),
            "total_events": int(total_events),
            "overspeeding_count": int(overspeed_cnt),
            "harsh_braking_count": int(harsh_brake_cnt),
            "rapid_accel_count": int(rapid_accel_cnt),
            "sharp_turning_count": int(sharp_turn_cnt),
            "excessive_idling_count": int(idle_cnt),
            "avg_speed": float(avg_speed),
            "max_speed": float(max_speed),
            "last_calculated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    scores_df = pd.DataFrame(driver_rows)
    return scores_df


def explain_safety_score(driver_id: str, scores_df: pd.DataFrame, events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns an academic and transparent breakdown of the safety score calculation.
    """
    row = scores_df[scores_df["driver_id"] == driver_id]
    if row.empty:
        return {"error": f"Driver {driver_id} not found."}

    d = row.iloc[0]
    return {
        "driver_id": driver_id,
        "base_score": 100.0,
        "final_score": d["safety_score"],
        "risk_category": d["risk_category"],
        "total_events": d["total_events"],
        "breakdown": {
            "overspeeding": {
                "count": d["overspeeding_count"],
                "base_penalty_pts": BASE_PENALTIES["overspeeding"],
                "total_deduction_approx": round(d["overspeeding_count"] * BASE_PENALTIES["overspeeding"], 1)
            },
            "harsh_braking": {
                "count": d["harsh_braking_count"],
                "base_penalty_pts": BASE_PENALTIES["harsh_braking"],
                "total_deduction_approx": round(d["harsh_braking_count"] * BASE_PENALTIES["harsh_braking"], 1)
            },
            "rapid_acceleration": {
                "count": d["rapid_accel_count"],
                "base_penalty_pts": BASE_PENALTIES["rapid_acceleration"],
                "total_deduction_approx": round(d["rapid_accel_count"] * BASE_PENALTIES["rapid_acceleration"], 1)
            },
            "sharp_turning": {
                "count": d["sharp_turning_count"],
                "base_penalty_pts": BASE_PENALTIES["sharp_turning"],
                "total_deduction_approx": round(d["sharp_turning_count"] * BASE_PENALTIES["sharp_turning"], 1)
            },
            "excessive_idling": {
                "count": d["excessive_idling_count"],
                "base_penalty_pts": BASE_PENALTIES["excessive_idling"],
                "total_deduction_approx": round(d["excessive_idling_count"] * BASE_PENALTIES["excessive_idling"], 1)
            }
        },
        "disclaimer": (
            "Notice: This is a project-defined heuristic scoring model developed for demonstration "
            "and educational purposes. It is not an ISO/commercial actuarial underwriting standard."
        )
    }
