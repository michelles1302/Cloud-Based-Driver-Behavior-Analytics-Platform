"""
Machine Learning module for Driver Behavior Analytics Platform.
Trains a Scikit-Learn Random Forest Classifier to assess and classify
driving risk levels (Safe, Moderate, High Risk) from telemetry feature aggregates.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import database
from telemetry_generator import generate_fleet_telemetry
from behavior_detector import detect_all_events
from safety_score import calculate_driver_safety_scores

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "driver_risk_model.joblib")

FEATURE_COLUMNS = [
    "avg_speed",
    "max_speed",
    "speed_std",
    "min_accel",
    "max_accel",
    "harsh_brake_count",
    "rapid_accel_count",
    "sharp_turn_count",
    "excessive_idle_duration",
    "avg_abs_steering"
]

SYNTHETIC_DATA_DISCLOSURE = (
    "DISCLOSURE: This machine learning model was trained on synthetic telematics simulation data. "
    "While the features represent authentic physical dynamics (speed, acceleration, braking, "
    "and steering angles), synthetic data has uniform noise distributions. In real-world telematics, "
    "models must be calibrated across diverse vehicle models, weather conditions, GPS inaccuracies, "
    "and road classifications."
)


def extract_trip_features(telemetry_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts statistical and event-frequency features per trip for ML model training and inference.
    """
    rows = []
    for trip_id, trip in telemetry_df.groupby("trip_id"):
        driver_id = trip["driver_id"].iloc[0]
        trip_events = events_df[events_df["trip_id"] == trip_id] if not events_df.empty else pd.DataFrame()

        avg_speed = float(trip["speed"].mean())
        max_speed = float(trip["speed"].max())
        speed_std = float(trip["speed"].std()) if len(trip) > 1 else 0.0
        min_accel = float(trip["acceleration"].min())
        max_accel = float(trip["acceleration"].max())
        avg_abs_steering = float(trip["steering_angle"].abs().mean())

        if not trip_events.empty:
            harsh_brake_count = int(len(trip_events[trip_events["event_type"] == "harsh_braking"]))
            rapid_accel_count = int(len(trip_events[trip_events["event_type"] == "rapid_acceleration"]))
            sharp_turn_count = int(len(trip_events[trip_events["event_type"] == "sharp_turning"]))
            idle_events = trip_events[trip_events["event_type"] == "excessive_idling"]
            idle_duration = float(idle_events["value"].sum()) if not idle_events.empty else 0.0
        else:
            harsh_brake_count = 0
            rapid_accel_count = 0
            sharp_turn_count = 0
            idle_duration = 0.0

        # Ground-truth heuristic risk label based on behavioral events & speed
        risk_score = 0
        risk_score += harsh_brake_count * 2.0
        risk_score += rapid_accel_count * 1.5
        risk_score += sharp_turn_count * 1.5
        if max_speed > 105.0:
            risk_score += 2.0
        if idle_duration > 40.0:
            risk_score += 1.0

        if risk_score == 0 and max_speed <= 85.0:
            risk_label = "Safe"
        elif risk_score <= 5.0 and max_speed <= 105.0:
            risk_label = "Moderate"
        else:
            risk_label = "High Risk"

        rows.append({
            "trip_id": trip_id,
            "driver_id": driver_id,
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "speed_std": round(speed_std, 2),
            "min_accel": round(min_accel, 2),
            "max_accel": round(max_accel, 2),
            "harsh_brake_count": harsh_brake_count,
            "rapid_accel_count": rapid_accel_count,
            "sharp_turn_count": sharp_turn_count,
            "excessive_idle_duration": round(idle_duration, 1),
            "avg_abs_steering": round(avg_abs_steering, 2),
            "risk_label": risk_label
        })

    return pd.DataFrame(rows)


def train_driver_risk_model(
    sample_drivers: int = 15,
    trips_per_driver: int = 4,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Generates training dataset, extracts features, trains Random Forest Classifier,
    evaluates on holdout test set, and saves model artifacts and metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Generating synthetic training telemetry ({sample_drivers} drivers, {trips_per_driver} trips/driver)...")

    # Generate dedicated training dataset
    train_telemetry = generate_fleet_telemetry(
        num_drivers=sample_drivers,
        trips_per_driver=trips_per_driver,
        trip_duration_seconds=120,
        seed=101
    )
    train_events = detect_all_events(train_telemetry)
    features_df = extract_trip_features(train_telemetry, train_events)

    X = features_df[FEATURE_COLUMNS]
    y = features_df["risk_label"]

    # Stratified train-test split if all classes have at least 2 members, else standard split
    can_stratify = len(y.unique()) > 1 and y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if can_stratify else None
    )

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    # Predict on holdout test set
    y_pred = rf_model.predict(X_test)

    # Calculate standard evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    classes = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Feature importances
    feature_importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_model.feature_importances_]))

    model_artifact = {
        "model": rf_model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": classes,
        "trained_at": datetime.now().isoformat(),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "samples_count": len(features_df),
            "test_samples": len(y_test)
        },
        "feature_importances": feature_importances,
        "confusion_matrix": cm.tolist(),
        "disclosure": SYNTHETIC_DATA_DISCLOSURE
    }

    if save_model:
        joblib.dump(model_artifact, MODEL_PATH)
        print(f"Model saved successfully to {MODEL_PATH}")

        # Record in database
        database.init_db()
        database.save_model_metric({
            "model_name": "Random Forest Risk Classifier",
            "trained_at": model_artifact["trained_at"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "samples_count": len(features_df),
            "notes": f"Classes: {', '.join(classes)} | Trained on {len(features_df)} simulated trips"
        })

    return {
        "model": rf_model,
        "metrics": model_artifact["metrics"],
        "confusion_matrix": cm,
        "classes": classes,
        "classification_report": clf_report,
        "feature_importances": feature_importances,
        "disclosure": SYNTHETIC_DATA_DISCLOSURE
    }


def load_driver_risk_model() -> Optional[Dict[str, Any]]:
    """Loads the saved Random Forest model and associated metadata."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        return None


def predict_driving_risk(feature_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Predicts driving risk category given an input dictionary of telemetry features.
    """
    artifact = load_driver_risk_model()
    if artifact is None:
        # Fallback heuristic if model is not yet trained
        return {
            "prediction": "Unknown (Model not trained)",
            "probabilities": {},
            "status": "error"
        }

    model = artifact["model"]
    classes = artifact["classes"]

    # Align features
    row = [feature_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    X_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    prediction = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]
    prob_dict = {cls: round(float(probs[i]), 3) for i, cls in enumerate(classes)}

    return {
        "prediction": prediction,
        "probabilities": prob_dict,
        "status": "success",
        "feature_columns": FEATURE_COLUMNS
    }


if __name__ == "__main__":
    print("Training Random Forest Driver Behavior Classifier...")
    results = train_driver_risk_model(sample_drivers=15, trips_per_driver=4, save_model=True)
    m = results["metrics"]
    print("\n--- Model Training Results ---")
    print(f"Accuracy:  {m['accuracy'] * 100:.2f}%")
    print(f"Precision: {m['precision'] * 100:.2f}%")
    print(f"Recall:    {m['recall'] * 100:.2f}%")
    print(f"F1 Score:  {m['f1_score'] * 100:.2f}%")
    print(f"Classes:   {results['classes']}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nFeature Importances:")
    for feat, imp in sorted(results["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:25s}: {imp:.4f}")
    print(f"\n{results['disclosure']}")
