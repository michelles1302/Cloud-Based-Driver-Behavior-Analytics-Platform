"""
End-to-End Test Suite for Cloud-Based Driver Behavior Analytics Platform.
Tests all core modules:
1. Telemetry Simulation
2. Unsafe Behavior Detection
3. Safety Scoring & Explanations
4. SQLite Persistence
5. Machine Learning Training & Inference
6. Cloud Storage (Local fallback handling)
"""

import os
import unittest
import pandas as pd
import numpy as np

import database
import telemetry_generator
import behavior_detector
import safety_score
import train_model
import cloud_storage
from pipeline import run_full_pipeline

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "test_driver_analytics.db")


class TestDriverAnalyticsPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use a temporary test database
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        database.init_db(TEST_DB_PATH)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    def test_01_telemetry_generator(self):
        """Test telemetry generator produces valid columns and realistic values."""
        df = telemetry_generator.generate_fleet_telemetry(
            num_drivers=3,
            trips_per_driver=2,
            trip_duration_seconds=30
        )
        self.assertFalse(df.empty)
        required_cols = [
            "driver_id", "vehicle_id", "trip_id", "timestamp",
            "speed", "acceleration", "braking_intensity", "steering_angle", "distance_travelled"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        self.assertEqual(df["driver_id"].nunique(), 3)
        self.assertEqual(df["trip_id"].nunique(), 6)
        self.assertTrue((df["speed"] >= 0.0).all())
        self.assertTrue((df["distance_travelled"] >= 0.0).all())

    def test_02_behavior_detector(self):
        """Test behavior detector identifies unsafe maneuvers and groups events."""
        # Create a synthetic dataframe with explicit violations
        synthetic_trip = pd.DataFrame([
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:00", "speed": 40.0, "acceleration": 1.0, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 0.01},
            # Overspeeding (>= 3s)
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:01", "speed": 115.0, "acceleration": 0.5, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 0.04},
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:02", "speed": 118.0, "acceleration": 0.2, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 0.07},
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:03", "speed": 112.0, "acceleration": 0.0, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 0.10},
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:04", "speed": 80.0, "acceleration": -1.0, "braking_intensity": 20.0, "steering_angle": 0.0, "distance_travelled": 0.12},
            # Harsh braking
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:05", "speed": 50.0, "acceleration": -3.8, "braking_intensity": 90.0, "steering_angle": 0.0, "distance_travelled": 0.13},
            # Rapid accel
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:06", "speed": 60.0, "acceleration": 3.6, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 0.15},
            # Sharp turn
            {"driver_id": "DRV-T1", "vehicle_id": "VEH-T1", "trip_id": "TRIP-T1", "timestamp": "2026-01-01 10:00:07", "speed": 45.0, "acceleration": 0.0, "braking_intensity": 0.0, "steering_angle": 34.0, "distance_travelled": 0.16},
        ])

        events = behavior_detector.detect_trip_events(synthetic_trip)
        self.assertGreater(len(events), 0)
        types_detected = {e["event_type"] for e in events}
        self.assertIn("overspeeding", types_detected)
        self.assertIn("harsh_braking", types_detected)
        self.assertIn("rapid_acceleration", types_detected)
        self.assertIn("sharp_turning", types_detected)

    def test_03_safety_score_calculation(self):
        """Test safety score bounds, risk categories, and transparency."""
        telemetry_df = pd.DataFrame([
            {"driver_id": "D1", "vehicle_id": "V1", "trip_id": "T1", "timestamp": "2026-01-01 10:00:00", "speed": 50.0, "acceleration": 0.0, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 5.0},
            {"driver_id": "D2", "vehicle_id": "V2", "trip_id": "T2", "timestamp": "2026-01-01 10:00:00", "speed": 95.0, "acceleration": 0.0, "braking_intensity": 0.0, "steering_angle": 0.0, "distance_travelled": 10.0}
        ])
        events_df = pd.DataFrame([
            {"event_id": "E1", "driver_id": "D2", "vehicle_id": "V2", "trip_id": "T2", "timestamp": "2026-01-01 10:00:01", "event_type": "overspeeding", "severity": "Medium", "value": 115.0, "description": "speeding"},
            {"event_id": "E2", "driver_id": "D2", "vehicle_id": "V2", "trip_id": "T2", "timestamp": "2026-01-01 10:00:02", "event_type": "harsh_braking", "severity": "High", "value": -4.2, "description": "brake"},
            {"event_id": "E3", "driver_id": "D2", "vehicle_id": "V2", "trip_id": "T2", "timestamp": "2026-01-01 10:00:03", "event_type": "rapid_acceleration", "severity": "Medium", "value": 3.8, "description": "accel"},
        ])

        scores_df = safety_score.calculate_driver_safety_scores(telemetry_df, events_df)
        self.assertEqual(len(scores_df), 2)

        # D1 has no events -> 100.0 Excellent
        d1 = scores_df[scores_df["driver_id"] == "D1"].iloc[0]
        self.assertEqual(d1["safety_score"], 100.0)
        self.assertEqual(d1["risk_category"], "Excellent")

        # D2 has 3 events -> lower score
        d2 = scores_df[scores_df["driver_id"] == "D2"].iloc[0]
        self.assertLess(d2["safety_score"], 100.0)

        # Transparency explanation
        exp = safety_score.explain_safety_score("D2", scores_df, events_df)
        self.assertIn("breakdown", exp)
        self.assertIn("disclaimer", exp)

    def test_04_database_operations(self):
        """Test SQLite persistence and aggregation queries."""
        df_telem = pd.DataFrame([{
            "driver_id": "DRV-DB1", "vehicle_id": "VEH-DB1", "trip_id": "TRIP-DB1",
            "timestamp": "2026-01-01 12:00:00", "speed": 45.0, "acceleration": 0.5,
            "braking_intensity": 0.0, "steering_angle": 1.2, "distance_travelled": 0.5
        }])
        database.save_telemetry(df_telem, db_path=TEST_DB_PATH)
        queried = database.get_telemetry(trip_id="TRIP-DB1", db_path=TEST_DB_PATH)
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried.iloc[0]["driver_id"], "DRV-DB1")

    def test_05_machine_learning_inference(self):
        """Test that the trained Random Forest model loads and makes valid predictions."""
        artifact = train_model.load_driver_risk_model()
        self.assertIsNotNone(artifact, "Random forest model artifact should exist in models/")
        self.assertIn("model", artifact)
        self.assertIn("metrics", artifact)

        # Test single prediction
        sample_features = {
            "avg_speed": 45.0,
            "max_speed": 60.0,
            "speed_std": 5.0,
            "min_accel": -1.5,
            "max_accel": 1.8,
            "harsh_brake_count": 0,
            "rapid_accel_count": 0,
            "sharp_turn_count": 0,
            "excessive_idle_duration": 0.0,
            "avg_abs_steering": 2.1
        }
        res = train_model.predict_driving_risk(sample_features)
        self.assertEqual(res["status"], "success")
        self.assertIn(res["prediction"], artifact["classes"])
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=1)

    def test_06_cloud_storage_local_mode(self):
        """Test cloud storage handles absent AWS credentials safely without crashing."""
        status = cloud_storage.test_s3_connection()
        self.assertIn("status", status)
        self.assertIn("message", status)
        # Attempting upload without credentials returns clean failure dict
        up_res = cloud_storage.upload_file_to_s3("non_existent_file.csv")
        self.assertFalse(up_res["success"])
        self.assertFalse(up_res["verified"])


if __name__ == "__main__":
    unittest.main()
