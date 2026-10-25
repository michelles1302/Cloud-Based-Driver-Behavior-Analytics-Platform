"""
Telemetry Data Simulator for Driver Behavior Analytics Platform.
Generates realistic driving telemetry for multiple drivers and trips with
distinct behavioral patterns (normal, overspeeding, harsh braking, rapid acceleration,
sharp turns, and excessive idling).
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Default configuration for driver archetypes
DRIVER_PROFILES = [
    {
        "driver_id": "DRV-001",
        "vehicle_id": "VEH-101",
        "name": "Alex Chen",
        "archetype": "safe_commuter",
        "overspeed_prob": 0.00,
        "harsh_brake_prob": 0.00,
        "rapid_accel_prob": 0.00,
        "sharp_turn_prob": 0.00,
        "idle_tendency": "low",
        "base_speed_range": (35, 60)
    },
    {
        "driver_id": "DRV-002",
        "vehicle_id": "VEH-102",
        "name": "Beatriz Silva",
        "archetype": "moderate_driver",
        "overspeed_prob": 0.015,
        "harsh_brake_prob": 0.010,
        "rapid_accel_prob": 0.010,
        "sharp_turn_prob": 0.010,
        "idle_tendency": "low",
        "base_speed_range": (45, 75)
    },
    {
        "driver_id": "DRV-003",
        "vehicle_id": "VEH-103",
        "name": "Carlos Diaz",
        "archetype": "speeding_enthusiast",
        "overspeed_prob": 0.050,
        "harsh_brake_prob": 0.015,
        "rapid_accel_prob": 0.025,
        "sharp_turn_prob": 0.015,
        "idle_tendency": "medium",
        "base_speed_range": (60, 110)
    },
    {
        "driver_id": "DRV-004",
        "vehicle_id": "VEH-104",
        "name": "David Miller",
        "archetype": "aggressive_courier",
        "overspeed_prob": 0.035,
        "harsh_brake_prob": 0.035,
        "rapid_accel_prob": 0.030,
        "sharp_turn_prob": 0.035,
        "idle_tendency": "medium",
        "base_speed_range": (40, 85)
    },
    {
        "driver_id": "DRV-005",
        "vehicle_id": "VEH-105",
        "name": "Elena Rostova",
        "archetype": "high_risk_idle",
        "overspeed_prob": 0.030,
        "harsh_brake_prob": 0.030,
        "rapid_accel_prob": 0.025,
        "sharp_turn_prob": 0.025,
        "idle_tendency": "high",
        "base_speed_range": (30, 95)
    }
]


def simulate_trip(
    driver_info: dict,
    trip_id: str,
    start_time: datetime,
    duration_seconds: int = 180,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulates a single driving trip with 1-second telemetry ticks.
    Realistic physical dynamics: speed, acceleration, braking, steering, distance.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    records = []
    current_time = start_time
    current_speed = 0.0  # start stationary
    cumulative_distance = 0.0  # in kilometers
    dt = 1.0  # 1-second interval

    # State tracking
    is_idling = False
    idle_counter = 0

    target_idle_duration = 0
    if driver_info["idle_tendency"] == "high":
        # Introduce an excessive idling block (e.g. 70-90s)
        idle_start_tick = random.randint(30, max(35, duration_seconds - 90))
        target_idle_duration = random.randint(70, 90)
    elif driver_info["idle_tendency"] == "medium":
        idle_start_tick = random.randint(40, max(45, duration_seconds - 70))
        target_idle_duration = random.randint(45, 65)
    else:
        idle_start_tick = -1
    # Maneuver and cooldown state tracking
    active_maneuver = None  # None, 'brake', 'accel', 'turn'
    maneuver_duration_left = 0
    cooldown_ticks_left = 10  # Start with smooth initial rollout

    for t in range(duration_seconds):
        # 1. Check if entering or in idling state
        if t == idle_start_tick and current_speed < 20.0:
            is_idling = True
            idle_counter = 0

        if is_idling:
            current_speed = 0.0
            acceleration = 0.0
            braking_intensity = 0.0
            steering_angle = 0.0
            idle_counter += 1
            if idle_counter >= target_idle_duration:
                is_idling = False
                cooldown_ticks_left = 5
        else:
            # Check ongoing maneuver
            if maneuver_duration_left > 0:
                maneuver_duration_left -= 1
                if active_maneuver == "brake":
                    acceleration = random.uniform(-4.2, -3.2)
                    braking_intensity = random.uniform(80.0, 95.0)
                    steering_angle = random.uniform(-4.0, 4.0)
                elif active_maneuver == "accel":
                    acceleration = random.uniform(3.2, 4.4)
                    braking_intensity = 0.0
                    steering_angle = random.uniform(-3.0, 3.0)
                elif active_maneuver == "turn":
                    acceleration = random.uniform(-0.8, 0.4)
                    braking_intensity = 15.0 if acceleration < 0 else 0.0
                    # Maintain sharp steering
                    steering_angle = turn_direction * random.uniform(29.0, 38.0)
            else:
                active_maneuver = None

                # Check if we can trigger a new maneuver
                if cooldown_ticks_left > 0:
                    cooldown_ticks_left -= 1
                    r_event = 1.0  # In cooldown, drive normally
                else:
                    r_event = random.random()

                # Trigger Harsh Braking Maneuver (lasts 2-3s)
                if r_event < driver_info["harsh_brake_prob"] and current_speed > 35.0:
                    active_maneuver = "brake"
                    maneuver_duration_left = random.randint(2, 3)
                    cooldown_ticks_left = random.randint(12, 20)
                    acceleration = random.uniform(-4.2, -3.2)
                    braking_intensity = random.uniform(80.0, 95.0)
                    steering_angle = random.uniform(-4.0, 4.0)

                # Trigger Rapid Acceleration Maneuver (lasts 2-3s)
                elif r_event < (driver_info["harsh_brake_prob"] + driver_info["rapid_accel_prob"]) and current_speed < 110.0:
                    active_maneuver = "accel"
                    maneuver_duration_left = random.randint(2, 3)
                    cooldown_ticks_left = random.randint(12, 18)
                    acceleration = random.uniform(3.2, 4.4)
                    braking_intensity = 0.0
                    steering_angle = random.uniform(-3.0, 3.0)

                # Trigger Sharp Turn Maneuver (lasts 2-3s)
                elif r_event < (driver_info["harsh_brake_prob"] + driver_info["rapid_accel_prob"] + driver_info["sharp_turn_prob"]) and current_speed > 30.0:
                    active_maneuver = "turn"
                    turn_direction = random.choice([-1, 1])
                    maneuver_duration_left = random.randint(2, 3)
                    cooldown_ticks_left = random.randint(12, 18)
                    acceleration = random.uniform(-0.8, 0.4)
                    braking_intensity = 15.0 if acceleration < 0 else 0.0
                    steering_angle = turn_direction * random.uniform(29.0, 38.0)

                # Normal driving or Overspeeding cruise
                else:
                    if random.random() < driver_info["overspeed_prob"]:
                        target_speed = random.uniform(105.0, 125.0)
                    else:
                        target_speed = random.uniform(driver_info["base_speed_range"][0], driver_info["base_speed_range"][1])

                    speed_diff = target_speed - current_speed
                    if speed_diff > 2.0:
                        acceleration = random.uniform(0.6, 2.0)
                        braking_intensity = 0.0
                    elif speed_diff < -2.0:
                        acceleration = random.uniform(-2.0, -0.6)
                        braking_intensity = random.uniform(15.0, 40.0)
                    else:
                        acceleration = random.uniform(-0.3, 0.3)
                        braking_intensity = 0.0

                    # Natural steering variations
                    steering_angle = float(np.random.normal(0.0, 3.0))

            # Update speed: v = u + a * dt (km/h)
            current_speed += acceleration * dt * 3.6
            current_speed = max(0.0, min(140.0, current_speed))

            if current_speed == 0.0:
                braking_intensity = 0.0

        # Incremental distance in km: (km/h) / 3600 * 1s
        incremental_distance = (current_speed / 3600.0) * dt
        cumulative_distance += incremental_distance

        records.append({
            "driver_id": driver_info["driver_id"],
            "vehicle_id": driver_info["vehicle_id"],
            "trip_id": trip_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "speed": round(current_speed, 2),
            "acceleration": round(acceleration, 2),
            "braking_intensity": round(braking_intensity, 2),
            "steering_angle": round(steering_angle, 2),
            "distance_travelled": round(cumulative_distance, 4)
        })

        current_time += timedelta(seconds=dt)

    return pd.DataFrame(records)


def generate_fleet_telemetry(
    num_drivers: int = 5,
    trips_per_driver: int = 2,
    trip_duration_seconds: int = 150,
    base_start_time: Optional[datetime] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a consolidated telemetry dataset for multiple drivers and trips.
    """
    if base_start_time is None:
        base_start_time = datetime.now() - timedelta(days=2)

    selected_profiles = []
    for i in range(num_drivers):
        base = DRIVER_PROFILES[i % len(DRIVER_PROFILES)]
        prof = dict(base)
        idx = i + 1
        prof["driver_id"] = f"DRV-{idx:03d}"
        prof["vehicle_id"] = f"VEH-{idx+100:03d}"
        prof["name"] = f"Driver {idx}" if i >= len(DRIVER_PROFILES) else base["name"]
        selected_profiles.append(prof)

    all_trips = []
    trip_counter = 1001

    for d_idx, driver in enumerate(selected_profiles):
        for trip_num in range(trips_per_driver):
            trip_id = f"TRIP-{trip_counter}"
            trip_start = base_start_time + timedelta(hours=(d_idx * 4) + (trip_num * 2), minutes=random.randint(5, 30))
            
            trip_df = simulate_trip(
                driver_info=driver,
                trip_id=trip_id,
                start_time=trip_start,
                duration_seconds=trip_duration_seconds,
                seed=seed + trip_counter
            )
            all_trips.append(trip_df)
            trip_counter += 1

    combined_df = pd.concat(all_trips, ignore_index=True)
    return combined_df


def save_telemetry_to_csv(df: pd.DataFrame, output_path: str = None) -> str:
    """Saves generated telemetry to CSV."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "data", "telemetry.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    print("Generating simulated driver telemetry dataset...")
    df = generate_fleet_telemetry(num_drivers=5, trips_per_driver=2, trip_duration_seconds=150)
    csv_path = save_telemetry_to_csv(df)
    print(f"Generated {len(df)} telemetry records across {df['driver_id'].nunique()} drivers and {df['trip_id'].nunique()} trips.")
    print(f"Saved dataset to {csv_path}")
    print("\nSample records:")
    print(df.head(5))
