"""
Cloud-Based Driver Behavior Analytics Platform
Interactive Streamlit Dashboard.
Provides Fleet Overview, Driver Analysis, Trip Analytics, Behavior Analytics,
Scikit-Learn ML Risk Predictor, and Optional Amazon S3 Cloud Storage Integration.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Local platform modules
import database
import telemetry_generator
import behavior_detector
import safety_score
import train_model
import cloud_storage
from pipeline import run_full_pipeline

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Driver Behavior Analytics Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished metrics and badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-excellent {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-good {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-highrisk {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_badge_html(category: str) -> str:
    """Returns styled HTML badge for risk category."""
    cat = (category or "").strip().lower()
    if "excellent" in cat:
        return f'<span class="badge-excellent">🛡️ {category}</span>'
    elif "good" in cat:
        return f'<span class="badge-good">✅ {category}</span>'
    elif "moderate" in cat:
        return f'<span class="badge-moderate">⚠️ {category}</span>'
    else:
        return f'<span class="badge-highrisk">🚨 {category}</span>'


# -----------------------------------------------------------------------------
# Data Initialization
# -----------------------------------------------------------------------------
database.init_db()

# Auto-seed database if empty
fleet_summary = database.get_fleet_summary()
if fleet_summary["total_drivers"] == 0:
    run_full_pipeline(num_drivers=5, trips_per_driver=2, trip_duration_seconds=150)
    fleet_summary = database.get_fleet_summary()

# -----------------------------------------------------------------------------
# Sidebar: Controls, Data Generation, Thresholds & Cloud Status
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/car.png", width=64)
st.sidebar.title("Telemetry Controls")

st.sidebar.markdown("### ⚡ Telemetry Generator")
sim_drivers = st.sidebar.slider("Number of Drivers", min_value=2, max_value=10, value=5, step=1)
sim_trips = st.sidebar.slider("Trips per Driver", min_value=1, max_value=4, value=2, step=1)
sim_duration = st.sidebar.slider("Trip Duration (seconds)", min_value=60, max_value=300, value=150, step=30)

if st.sidebar.button("⚡ Generate New Data", use_container_width=True, type="primary"):
    with st.spinner("Simulating multi-driver telemetry & detecting events..."):
        res = run_full_pipeline(
            num_drivers=sim_drivers,
            trips_per_driver=sim_trips,
            trip_duration_seconds=sim_duration
        )
        st.sidebar.success(f"Generated {res['telemetry_count']} points ({res['events_count']} events)!")
        st.rerun()

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
# Configurable Thresholds Expander
with st.sidebar.expander("⚙️ Detection Thresholds", expanded=False):
    th_speed = st.number_input("Speed Limit (km/h)", value=100.0, step=5.0)
    th_harsh_brake = st.number_input("Harsh Brake Accel (m/s²)", value=-3.0, step=0.5, max_value=-1.0)
    th_rapid_accel = st.number_input("Rapid Accel Spike (m/s²)", value=3.0, step=0.5, min_value=1.0)
    th_sharp_turn = st.number_input("Sharp Turn Angle (°)", value=28.0, step=2.0)
    th_idle_time = st.number_input("Excessive Idle (seconds)", value=60, step=10)

    if st.button("Apply & Re-detect Events", use_container_width=True):
        custom_cfg = {
            "speed_limit_kmh": th_speed,
            "harsh_brake_accel_ms2": th_harsh_brake,
            "rapid_accel_ms2": th_rapid_accel,
            "sharp_turn_angle_deg": th_sharp_turn,
            "excessive_idle_seconds": th_idle_time
        }
        raw_telemetry = database.get_telemetry()
        if not raw_telemetry.empty:
            with st.spinner("Re-evaluating events with updated thresholds..."):
                events_df = behavior_detector.detect_all_events(raw_telemetry, thresholds=custom_cfg)
                database.save_events(events_df, if_exists="replace")
                scores_df = safety_score.calculate_driver_safety_scores(raw_telemetry, events_df)
                database.save_driver_scores(scores_df)
                st.success("Events and safety scores updated!")
                st.rerun()

st.sidebar.markdown("---")
# Optional Cloud S3 Status Indicator
st.sidebar.markdown("### ☁️ Cloud Sync (Amazon S3)")
s3_configured = cloud_storage.is_s3_configured()
if s3_configured:
    cfg = cloud_storage.get_s3_config()
    st.sidebar.success(f"Configured: `{cfg['s3_bucket_name']}`")
    if st.sidebar.button("☁️ Upload Telemetry to S3", use_container_width=True):
        with st.spinner("Uploading telemetry CSV to Amazon S3..."):
            up_res = cloud_storage.backup_telemetry_to_s3()
            if up_res["success"]:
                st.sidebar.success(f"Verified {up_res['size_bytes']:,} bytes in S3!")
                st.sidebar.info(f"URI: `{up_res['s3_uri']}`")
            else:
                st.sidebar.error(up_res["message"])
else:
    st.sidebar.info("🌐 Running in Local Mode (AWS credentials not set).")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🚗 Cloud-Based Driver Behavior Analytics Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">End-to-End IoT Telemetry Simulation • Rule-Based Unsafe Event Detection • '
    'Driver Safety Scoring • Scikit-Learn Machine Learning • Optional S3 Cloud Archive</div>',
    unsafe_allow_html=True
)

# Fetch latest database records
fleet_summary = database.get_fleet_summary()
driver_scores_df = database.get_driver_scores()
events_df = database.get_events()
telemetry_df = database.get_telemetry()

# Tabs matching user requirements
tab_overview, tab_driver, tab_trip, tab_behavior, tab_ml, tab_cloud = st.tabs([
    "📊 Fleet Overview",
    "👤 Driver Analysis",
    "🛣️ Trip Analytics",
    "⚠️ Behavior Analytics",
    "🤖 Machine Learning",
    "☁️ Cloud Integration"
])


# =============================================================================
# 1. FLEET OVERVIEW TAB
# =============================================================================
with tab_overview:
    st.subheader("Fleet Performance & Safety Overview")

    # KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="👥 Total Drivers", value=fleet_summary["total_drivers"])
    with c2:
        st.metric(label="🛣️ Total Trips Recorded", value=fleet_summary["total_trips"])
    with c3:
        avg_score = fleet_summary["avg_safety_score"]
        st.metric(label="🛡️ Fleet Avg Safety Score", value=f"{avg_score:.1f} / 100")
    with c4:
        st.metric(label="⚠️ Total Unsafe Events", value=f"{fleet_summary['total_unsafe_events']:,}")

    st.markdown("---")

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.markdown("#### Driver Safety Scores Comparison")
        if not driver_scores_df.empty:
            # Color map based on risk category
            color_map = {
                "Excellent": "#10B981",
                "Good": "#3B82F6",
                "Moderate": "#F59E0B",
                "High Risk": "#EF4444"
            }
            fig_scores = px.bar(
                driver_scores_df,
                x="driver_id",
                y="safety_score",
                color="risk_category",
                color_discrete_map=color_map,
                text="safety_score",
                labels={"safety_score": "Safety Score (0-100)", "driver_id": "Driver ID", "risk_category": "Category"},
                title="Fleet Safety Scores by Driver"
            )
            fig_scores.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_scores.update_layout(yaxis_range=[0, 110], height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.info("No driver scores available.")

    with col_table:
        st.markdown("#### Driver Fleet Summary Table")
        if not driver_scores_df.empty:
            display_df = driver_scores_df[[
                "driver_id", "safety_score", "risk_category", "total_trips",
                "total_events", "avg_speed", "max_speed", "total_distance_km"
            ]].copy()
            display_df.rename(columns={
                "driver_id": "Driver ID",
                "safety_score": "Score",
                "risk_category": "Risk Category",
                "total_trips": "Trips",
                "total_events": "Events",
                "avg_speed": "Avg Speed (km/h)",
                "max_speed": "Max Speed (km/h)",
                "total_distance_km": "Distance (km)"
            }, inplace=True)
            st.dataframe(display_df, use_container_width=True, height=350)
        else:
            st.info("No driver score data available.")


# =============================================================================
# 2. DRIVER ANALYSIS TAB
# =============================================================================
with tab_driver:
    st.subheader("Individual Driver Safety & Telematics Profile")

    if driver_scores_df.empty:
        st.warning("No driver records found. Please generate sample telemetry from the sidebar.")
    else:
        driver_list = list(driver_scores_df["driver_id"].unique())
        selected_driver = st.selectbox("Select Driver to Inspect:", driver_list, index=0)

        d_row = driver_scores_df[driver_scores_df["driver_id"] == selected_driver].iloc[0]
        d_events = events_df[events_df["driver_id"] == selected_driver] if not events_df.empty else pd.DataFrame()
        d_telemetry = telemetry_df[telemetry_df["driver_id"] == selected_driver] if not telemetry_df.empty else pd.DataFrame()

        # Driver Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Safety Score", f"{d_row['safety_score']} / 100")
        with m2:
            st.markdown(f"**Risk Category**<br>{get_risk_badge_html(d_row['risk_category'])}", unsafe_allow_html=True)
        with m3:
            st.metric("Average Speed", f"{d_row['avg_speed']} km/h")
        with m4:
            st.metric("Maximum Speed", f"{d_row['max_speed']} km/h")
        with m5:
            st.metric("Total Unsafe Events", int(d_row["total_events"]))

        st.markdown("---")

        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown("#### Unsafe Event Counts by Type")
            event_breakdown = pd.DataFrame({
                "Event Type": ["Overspeeding", "Harsh Braking", "Rapid Accel", "Sharp Turning", "Excessive Idling"],
                "Count": [
                    int(d_row["overspeeding_count"]),
                    int(d_row["harsh_braking_count"]),
                    int(d_row["rapid_accel_count"]),
                    int(d_row["sharp_turning_count"]),
                    int(d_row["excessive_idling_count"])
                ]
            })
            fig_dev = px.bar(
                event_breakdown,
                x="Event Type",
                y="Count",
                color="Event Type",
                title=f"Detected Events for {selected_driver}",
                text="Count"
            )
            fig_dev.update_traces(textposition='outside')
            fig_dev.update_layout(height=340, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dev, use_container_width=True)

        with d_col2:
            st.markdown("#### Transparent Safety Score Calculation Breakdown")
            exp = safety_score.explain_safety_score(selected_driver, driver_scores_df, events_df)
            st.markdown(f"""
            - **Base Starting Score:** 100.0 pts
            - **Overspeeding Deductions:** {exp['breakdown']['overspeeding']['count']} events (-{exp['breakdown']['overspeeding']['total_deduction_approx']} pts)
            - **Harsh Braking Deductions:** {exp['breakdown']['harsh_braking']['count']} events (-{exp['breakdown']['harsh_braking']['total_deduction_approx']} pts)
            - **Rapid Acceleration Deductions:** {exp['breakdown'].get('rapid_accel', {}).get('count', 0)} events (-{exp['breakdown'].get('rapid_accel', {}).get('total_deduction_approx', 0)} pts)
            - **Sharp Turning Deductions:** {exp['breakdown']['sharp_turning']['count']} events (-{exp['breakdown']['sharp_turning']['total_deduction_approx']} pts)
            - **Excessive Idling Deductions:** {exp['breakdown']['excessive_idling']['count']} events (-{exp['breakdown']['excessive_idling']['total_deduction_approx']} pts)
            - **Normalized Final Score:** `{exp['final_score']} / 100` ({exp['risk_category']})
            """)
            st.caption(f"ℹ️ {exp['disclaimer']}")

        # Trips table for this driver
        if not d_telemetry.empty:
            st.markdown("#### Trips Conducted by Driver")
            trips_agg = d_telemetry.groupby("trip_id").agg({
                "timestamp": ["min", "max"],
                "speed": ["mean", "max"],
                "distance_travelled": "max"
            }).reset_index()
            trips_agg.columns = ["Trip ID", "Start Time", "End Time", "Avg Speed (km/h)", "Max Speed (km/h)", "Distance (km)"]
            trips_agg["Avg Speed (km/h)"] = trips_agg["Avg Speed (km/h)"].round(1)
            trips_agg["Max Speed (km/h)"] = trips_agg["Max Speed (km/h)"].round(1)
            trips_agg["Distance (km)"] = trips_agg["Distance (km)"].round(2)
            st.dataframe(trips_agg, use_container_width=True)


# =============================================================================
# 3. TRIP ANALYTICS TAB
# =============================================================================
with tab_trip:
    st.subheader("Trip Telemetry & Sensor Dynamics")

    if telemetry_df.empty:
        st.warning("No telemetry records available.")
    else:
        trip_list = list(telemetry_df["trip_id"].unique())
        c_sel1, c_sel2 = st.columns([1, 2])
        with c_sel1:
            selected_trip = st.selectbox("Select Trip ID:", trip_list, index=0)

        trip_telemetry = telemetry_df[telemetry_df["trip_id"] == selected_trip].sort_values("timestamp")
        trip_events = events_df[events_df["trip_id"] == selected_trip] if not events_df.empty else pd.DataFrame()

        t_driver = trip_telemetry["driver_id"].iloc[0]
        t_vehicle = trip_telemetry["vehicle_id"].iloc[0]
        t_dist = trip_telemetry["distance_travelled"].max()
        t_dur = len(trip_telemetry)

        with c_sel2:
            st.markdown(
                f"**Driver:** `{t_driver}` | **Vehicle:** `{t_vehicle}` | "
                f"**Duration:** `{t_dur}s` | **Distance:** `{t_dist:.2f} km` | "
                f"**Events on Trip:** `{len(trip_events)}`"
            )

        st.markdown("---")

        # 1. Speed vs Time Chart
        st.markdown("#### 1. Speed versus Time")
        fig_speed = px.line(
            trip_telemetry,
            x="timestamp",
            y="speed",
            title=f"Trip {selected_trip} - Speed Profile (km/h)",
            labels={"speed": "Speed (km/h)", "timestamp": "Timestamp"}
        )
        # Add configured Speed Limit Line
        fig_speed.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Speed Limit (100 km/h)",
            annotation_position="top left"
        )
        fig_speed.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_speed, use_container_width=True)

        # 2. Acceleration vs Time Chart
        st.markdown("#### 2. Acceleration & Deceleration versus Time")
        fig_accel = px.line(
            trip_telemetry,
            x="timestamp",
            y="acceleration",
            title=f"Trip {selected_trip} - Acceleration Profile (m/s²)",
            labels={"acceleration": "Acceleration (m/s²)", "timestamp": "Timestamp"},
            color_discrete_sequence=["#0284C7"]
        )
        # Add Harsh Braking Threshold Line
        fig_accel.add_hline(
            y=-3.0,
            line_dash="dash",
            line_color="crimson",
            annotation_text="Harsh Brake Threshold (-3.0 m/s²)",
            annotation_position="bottom left"
        )
        # Add Rapid Acceleration Threshold Line
        fig_accel.add_hline(
            y=3.0,
            line_dash="dash",
            line_color="orange",
            annotation_text="Rapid Accel Threshold (+3.0 m/s²)",
            annotation_position="top left"
        )
        fig_accel.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_accel, use_container_width=True)

        # 3. Driving Events Summary for this trip
        st.markdown("#### 3. Driving Events Detected on Trip")
        if not trip_events.empty:
            show_ev = trip_events[[
                "event_id", "timestamp", "event_type", "severity", "value", "description"
            ]].copy()
            st.dataframe(show_ev, use_container_width=True)
        else:
            st.success("✅ Clean trip! No unsafe events detected for this journey.")


# =============================================================================
# 4. BEHAVIOR ANALYTICS TAB
# =============================================================================
with tab_behavior:
    st.subheader("Fleet-Wide Behavioral Risk & Distribution Analytics")

    if events_df.empty:
        st.info("No unsafe events recorded yet. Generate telemetry to populate behavior analytics.")
    else:
        b_c1, b_c2 = st.columns(2)

        with b_c1:
            st.markdown("#### 1. Unsafe Event Counts by Type")
            type_counts = events_df["event_type"].value_counts().reset_index()
            type_counts.columns = ["Event Type", "Total Events"]
            fig_types = px.pie(
                type_counts,
                names="Event Type",
                values="Total Events",
                hole=0.35,
                color="Event Type",
                title="Event Type Breakdown across Fleet"
            )
            fig_types.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_types, use_container_width=True)

        with b_c2:
            st.markdown("#### 2. Risk Category Distribution")
            if not driver_scores_df.empty:
                cat_counts = driver_scores_df["risk_category"].value_counts().reset_index()
                cat_counts.columns = ["Risk Category", "Driver Count"]
                color_map_cat = {
                    "Excellent": "#10B981",
                    "Good": "#3B82F6",
                    "Moderate": "#F59E0B",
                    "High Risk": "#EF4444"
                }
                fig_cats = px.pie(
                    cat_counts,
                    names="Risk Category",
                    values="Driver Count",
                    hole=0.4,
                    color="Risk Category",
                    color_discrete_map=color_map_cat,
                    title="Driver Classification Distribution"
                )
                fig_cats.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cats, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 3. Driver Safety Score Comparison")
        if not driver_scores_df.empty:
            fig_comp = px.bar(
                driver_scores_df.sort_values("safety_score", ascending=True),
                x="safety_score",
                y="driver_id",
                orientation="h",
                color="risk_category",
                color_discrete_map={
                    "Excellent": "#10B981",
                    "Good": "#3B82F6",
                    "Moderate": "#F59E0B",
                    "High Risk": "#EF4444"
                },
                text="safety_score",
                title="Driver Safety Scores (Ranked)",
                labels={"safety_score": "Safety Score (0-100)", "driver_id": "Driver"}
            )
            fig_comp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_comp.update_layout(xaxis_range=[0, 110], height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)


# =============================================================================
# 5. MACHINE LEARNING MODEL TAB
# =============================================================================
with tab_ml:
    st.subheader("Machine Learning: Driving Risk Classifier (Random Forest)")

    ml_artifact = train_model.load_driver_risk_model()

    if ml_artifact is None:
        st.warning("Model not found in `models/`. Click below to train the Random Forest Classifier.")
        if st.button("🤖 Train Random Forest Classifier Now", type="primary"):
            with st.spinner("Training Random Forest Classifier on simulated telemetry..."):
                train_res = train_model.train_driver_risk_model()
                st.success("Model trained and saved to `models/driver_risk_model.joblib`!")
                st.rerun()
    else:
        m = ml_artifact["metrics"]

        # Metric KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Test Accuracy", f"{m['accuracy'] * 100:.1f}%")
        with k2:
            st.metric("Precision (Weighted)", f"{m['precision'] * 100:.1f}%")
        with k3:
            st.metric("Recall (Weighted)", f"{m['recall'] * 100:.1f}%")
        with k4:
            st.metric("F1-Score", f"{m['f1_score'] * 100:.1f}%")

        st.markdown("---")

        ml_c1, ml_c2 = st.columns(2)

        with ml_c1:
            st.markdown("#### Confusion Matrix")
            classes = ml_artifact["classes"]
            cm_data = np.array(ml_artifact["confusion_matrix"])
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                x=classes,
                y=classes,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Risk Level", y="Actual Risk Level", color="Samples"),
                title="Holdout Test Set Confusion Matrix"
            )
            fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with ml_c2:
            st.markdown("#### Feature Importance")
            importances = ml_artifact.get("feature_importances", {})
            imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importance",
                color="Importance",
                color_continuous_scale="Teal"
            )
            fig_imp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")

        # Interactive Risk Predictor on New Sample Data
        st.markdown("#### 🧪 Test Driving Behavior Predictor on New Data")
        st.markdown("Adjust driving telemetry parameters below to test the trained model's real-time risk classification:")

        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            inp_avg_speed = st.slider("Average Speed (km/h)", 20.0, 130.0, 65.0, 1.0)
            inp_max_speed = st.slider("Max Speed (km/h)", 30.0, 150.0, 85.0, 1.0)
            inp_speed_std = st.slider("Speed Std Dev", 2.0, 40.0, 12.0, 1.0)
        with p_c2:
            inp_min_accel = st.slider("Max Decel / Min Accel (m/s²)", -5.5, 0.0, -1.8, 0.1)
            inp_max_accel = st.slider("Max Acceleration (m/s²)", 0.5, 5.0, 2.2, 0.1)
            inp_harsh_brakes = st.number_input("Harsh Braking Events", 0, 15, 1)
        with p_c3:
            inp_rapid_accels = st.number_input("Rapid Accel Events", 0, 15, 1)
            inp_sharp_turns = st.number_input("Sharp Turn Events", 0, 15, 0)
            inp_idling = st.slider("Excessive Idle Time (s)", 0.0, 120.0, 15.0, 5.0)

        input_sample = {
            "avg_speed": inp_avg_speed,
            "max_speed": inp_max_speed,
            "speed_std": inp_speed_std,
            "min_accel": inp_min_accel,
            "max_accel": inp_max_accel,
            "harsh_brake_count": inp_harsh_brakes,
            "rapid_accel_count": inp_rapid_accels,
            "sharp_turn_count": inp_sharp_turns,
            "excessive_idle_duration": inp_idling,
            "avg_abs_steering": 3.5
        }

        if st.button("🔮 Predict Driving Risk Class", type="primary"):
            pred_res = train_model.predict_driving_risk(input_sample)
            if pred_res["status"] == "success":
                pred_label = pred_res["prediction"]
                probs = pred_res["probabilities"]

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.markdown("### Model Prediction:")
                    st.markdown(get_risk_badge_html(pred_label), unsafe_allow_html=True)
                with r2:
                    st.markdown("### Class Probabilities:")
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Risk Class", "Probability"])
                    fig_pr = px.bar(
                        prob_df,
                        x="Risk Class",
                        y="Probability",
                        color="Risk Class",
                        text="Probability"
                    )
                    fig_pr.update_layout(yaxis_range=[0, 1.1], height=220)
                    st.plotly_chart(fig_pr, use_container_width=True)

        st.info(f"ℹ️ {ml_artifact['disclosure']}")


# =============================================================================
# 6. CLOUD INTEGRATION TAB (AMAZON S3)
# =============================================================================
with tab_cloud:
    st.subheader("Cloud Storage Architecture: Amazon S3 Integration")

    conn_status = cloud_storage.test_s3_connection()

    col_s3_info, col_s3_actions = st.columns([1, 1])

    with col_s3_info:
        st.markdown("#### Cloud Storage Status")
        if conn_status["status"] == "connected":
            st.success(f"✅ {conn_status['message']}")
            st.markdown(f"""
            - **Target Bucket:** `{conn_status['bucket']}`
            - **AWS Region:** `{conn_status['region']}`
            - **Service:** Amazon S3 Standard Storage
            """)
        elif conn_status["status"] == "local_mode":
            st.warning(f"🌐 **Local Mode Active**: {conn_status['message']}")
            st.info(f"👉 **Next Step to Enable Cloud**: {conn_status['guide']}")
        else:
            st.error(f"❌ **Connection Error**: {conn_status['message']}")

        st.markdown("#### Amazon S3 Setup & Pricing Guide")
        st.markdown(f"""
        - **Service:** `{cloud_storage.AWS_S3_DOCUMENTATION['service']}`
        - **Required IAM Permissions:** `{'`, `'.join(cloud_storage.AWS_S3_DOCUMENTATION['required_iam_permissions'])}`
        - **Pricing Notice:** {cloud_storage.AWS_S3_DOCUMENTATION['pricing_summary']}
        """)

    with col_s3_actions:
        st.markdown("#### S3 Cloud Backup Action")
        st.markdown(
            "Archive historical driving telemetry data to your remote Amazon S3 bucket. "
            "Data is exported to CSV and transmitted over encrypted HTTPS."
        )

        csv_file = os.path.join(os.path.dirname(__file__), "data", "telemetry.csv")
        csv_exists = os.path.exists(csv_file)

        if not csv_exists:
            st.warning("Telemetry CSV file not found on disk. Please generate sample data first.")

        if st.button("🚀 Upload Local Telemetry to S3", disabled=not csv_exists, type="primary"):
            with st.spinner("Connecting to Amazon S3 and uploading telemetry archive..."):
                result = cloud_storage.backup_telemetry_to_s3()
                if result["success"] and result["verified"]:
                    st.success(f"✅ Upload Verified! {result['message']}")
                    st.code(f"S3 URI: {result['s3_uri']}\nSize: {result['size_bytes']} bytes")
                else:
                    st.error(f"❌ Upload Failed: {result['message']}")
                    if "error_code" in result:
                        st.caption(f"Error Code: {result['error_code']}")

        st.markdown("---")
        st.markdown("#### Environment Setup Template (`.env`)")
        st.markdown("To enable AWS S3, place a `.env` file in the project root containing:")
        st.code("""
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=my-driver-analytics-bucket
        """, language="bash")
        st.caption("🔒 Security Reminder: Never commit real AWS credentials or keys to GitHub or public repositories.")
