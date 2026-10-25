# 🚗 Cloud-Based Driver Behavior Analytics Platform

An end-to-end telematics analytics platform built with Python, Pandas, Scikit-learn, Streamlit, Plotly, SQLite, and optional Amazon S3 cloud storage.

This project simulates IoT vehicular sensor telemetry, automatically detects unsafe driving behaviors using rule-based threshold heuristics, computes transparent driver safety scores, classifies trip risks using a trained Random Forest model, and provides a multi-tab interactive Streamlit dashboard.

---

## 📋 Table of Contents
1. [Key Features](#-key-features)
2. [Technology Stack](#-technology-stack)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [How It Works](#-how-it-works)
   - [A. Telemetry Simulation](#a-driver-telemetry-simulator)
   - [B. Unsafe Behavior Detection](#b-unsafe-behavior-detection)
   - [C. Driver Safety Score Formula](#c-driver-safety-score-formula)
   - [D. Machine Learning Risk Model](#d-machine-learning-risk-classifier)
   - [E. Streamlit Dashboard Guide](#e-streamlit-dashboard-guide)
7. [Cloud Integration (Amazon S3)](#-cloud-integration-amazon-s3)
8. [Running Tests](#-running-tests)
9. [Presentation & Demo Guide](#-presentation--demo-guide)
10. [Limitations & Disclosures](#-limitations--disclosures)

---

## 🌟 Key Features

- **Realistic Telematics Simulator**: Generates synchronized multi-driver vehicular telemetry (Speed, Acceleration, Braking Intensity, Steering Angle, Distance).
- **Rule-Based Unsafe Driving Detection**: Automatically flags 5 distinct hazardous driving behaviors:
  1. *Overspeeding*
  2. *Harsh Braking*
  3. *Rapid Acceleration*
  4. *Sharp Turning*
  5. *Excessive Idling*
- **Transparent Safety Scoring**: Computes an explainable 0–100 safety score and classifies drivers into four risk tiers: **Excellent**, **Good**, **Moderate**, and **High Risk**.
- **Scikit-Learn Machine Learning**: A trained Random Forest Classifier predicts driving risk classes with evaluated Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
- **Interactive Streamlit Web Dashboard**: Built-in KPI metrics, Plotly interactive timeseries plots, and threshold tuning sliders.
- **SQLite Local Persistence**: Works 100% offline without mandatory external cloud infrastructure.
- **Optional AWS S3 Cloud Storage**: Securely archives historical telemetry archives to Amazon S3 via Boto3 with verified upload confirmation.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core data simulation, ML modeling, and pipeline execution |
| **Data Processing** | Pandas, NumPy | Timeseries manipulation, event aggregation, and metrics |
| **Database** | SQLite 3 | Zero-configuration local relational persistence |
| **Machine Learning** | Scikit-Learn, Joblib | Random Forest model training, feature extraction, evaluation |
| **Dashboard** | Streamlit | Clean, interactive web UI with real-time controls |
| **Visualization** | Plotly | Dynamic charts (speed/accel timeseries, confusion matrix, bar plots) |
| **Cloud Storage** | Amazon S3 + Boto3 | Optional secure remote telematics archive |

*No heavy dependencies like Kafka, Kinesis, Docker, or Kubernetes are required.*

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[Telemetry Generator\ntelemetry_generator.py] -->|Raw 1s Ticks| B[Local SQLite Database\ndata/driver_analytics.db]
    B -->|Time-Series Query| C[Behavior Detector\nbehavior_detector.py]
    C -->|Detected Events| B
    B -->|Events & Distance| D[Safety Scoring Engine\nsafety_score.py]
    D -->|0-100 Scores & Tiers| B
    B -->|Aggregated Trip Features| E[ML Risk Classifier\ntrain_model.py]
    E -->|Model Artifact| F[models/driver_risk_model.joblib]
    
    B --> G[Streamlit Dashboard\napp.py]
    F --> G
    
    B -.->|Optional Telemetry CSV Export| H[Boto3 S3 Client\ncloud_storage.py]
    H -.->|HTTPS Upload| I[(Amazon S3 Bucket)]
```

---

## 📁 Project Structure

```text
cloud-driver-behavior-platform/
│
├── app.py                     # Streamlit interactive multi-tab dashboard
├── telemetry_generator.py     # Physics-based telematics simulation engine
├── behavior_detector.py       # Threshold-based unsafe event detection rules
├── safety_score.py            # Heuristic 0-100 scoring algorithm & transparency
├── train_model.py             # Random Forest ML model training & evaluation
├── database.py                # SQLite schema management, inserts, and queries
├── cloud_storage.py           # Optional Amazon S3 upload & verification (Boto3)
├── pipeline.py                # End-to-end simulation & database sync pipeline
├── test_platform.py           # Automated unit and integration test suite
│
├── requirements.txt           # Python dependencies
├── .env.example               # Template for optional AWS S3 credentials
├── .gitignore                 # Excludes caches, databases, and secrets
├── README.md                  # Complete documentation and presentation guide
│
├── data/
│   ├── driver_analytics.db    # SQLite database (auto-generated)
│   ├── telemetry.csv          # Exported telematics dataset
│   └── .gitkeep
│
└── models/
    ├── driver_risk_model.joblib # Trained Random Forest model artifact
    └── .gitkeep
```

---

## 🚀 Installation & Setup

### 1. Clone or Open Workspace
Open a terminal in the project directory:
```bash
cd "d:\Projects\CC\4"
```

### 2. (Optional) Create a Python Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Data & Train Machine Learning Model
Generate the initial telemetry dataset and train the Random Forest model:
```bash
python pipeline.py
python train_model.py
```

### 5. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔍 How It Works

### A. Driver Telemetry Simulator
The simulator (`telemetry_generator.py`) generates physical timeseries samples at 1-second intervals for multiple drivers and trips.
- **Speed ($v$)**: Modeled continuously in km/h based on acceleration ($v_{t} = v_{t-1} + a \cdot \Delta t \cdot 3.6$).
- **Acceleration ($a$)**: Deceleration (negative) and acceleration (positive) in m/s².
- **Braking Intensity**: Calculated proportionally when deceleration occurs (0% to 100%).
- **Steering Angle**: Modeled with realistic natural yaw deviations (-45° to +45°).
- **Distance Travelled**: Cumulative distance per trip in kilometers.

Simulated driver archetypes include:
- `Alex Chen` (Safe Commuter): Consistent speeds, gentle maneuvers.
- `Beatriz Silva` (Moderate Driver): Occasional mild speeding, mostly compliant.
- `Carlos Diaz` (Speeding Enthusiast): Frequent highway speeding bursts.
- `David Miller` (Aggressive Courier): Rapid accelerations, sharp turns, and sudden braking.
- `Elena Rostova` (High Risk / Idle): Prolonged stationary idling and erratic deceleration.

---

### B. Unsafe Behavior Detection
The detector (`behavior_detector.py`) applies configurable threshold heuristics and clusters contiguous sensor spikes into distinct maneuvers:

| Unsafe Behavior | Trigger Threshold | Severity Tiers (Low / Med / High) |
|---|---|---|
| **Overspeeding** | Speed > `100.0 km/h` sustained $\ge 3$s | < 112 / 112–125 / > 125 km/h |
| **Harsh Braking** | Deceleration $\le -3.0\text{ m/s}^2$ ($v > 10$ km/h) | -3.0 to -3.6 / -3.6 to -4.4 / < -4.4 m/s² |
| **Rapid Acceleration** | Acceleration $\ge +3.0\text{ m/s}^2$ | 3.0 to 3.6 / 3.6 to 4.3 / > 4.3 m/s² |
| **Sharp Turning** | Steering Angle $\ge |28.0^\circ|$ ($v \ge 25$ km/h) | 28° to 33° / 33° to 39° / > 39° |
| **Excessive Idling** | Speed $= 0$ km/h for $\ge 60$ seconds | 60–75s / 75–90s / > 90s |

---

### C. Driver Safety Score Formula
The safety scoring module (`safety_score.py`) implements a transparent penalty-deduction formula:

$$\text{Total Penalty} = \sum_{\text{events}} \left( \text{Base Penalty} \times \text{Severity Multiplier} \right)$$

$$\text{Normalized Penalty} = \frac{\text{Total Penalty}}{\max(1, \text{Total Trips})}$$

$$\text{Safety Score} = \max\left(0, \min\left(100, 100 - \text{Normalized Penalty}\right)\right)$$

#### Penalty Deductions:
- **Overspeeding**: 4.5 pts
- **Harsh Braking**: 5.0 pts
- **Rapid Acceleration**: 3.0 pts
- **Sharp Turning**: 3.5 pts
- **Excessive Idling**: 2.5 pts
- **Severity Multipliers**: Low = `0.8x`, Medium = `1.0x`, High = `1.3x`.

#### Risk Classification:
- **90 – 100**: 🛡️ **Excellent** (Careful, law-abiding driver)
- **75 – 89**: ✅ **Good** (Minor infractions, generally safe)
- **60 – 74**: ⚠️ **Moderate** (Frequent infractions, requires coaching)
- **Below 60**: 🚨 **High Risk** (Severe safety violations, intervention needed)

---

### D. Machine Learning Risk Classifier
The ML module (`train_model.py`) trains a Scikit-Learn **Random Forest Classifier** ($100$ estimators, max depth $6$) on statistical trip aggregates:
- **Feature Set**: `avg_speed`, `max_speed`, `speed_std`, `min_accel`, `max_accel`, `harsh_brake_count`, `rapid_accel_count`, `sharp_turn_count`, `excessive_idle_duration`, `avg_abs_steering`.
- **Target Classes**: `Safe`, `Moderate`, `High Risk`.
- **Evaluation**: 80/20 train/test split evaluated with:
  - Accuracy: **~80% – 87%**
  - Weighted Precision & Recall
  - Macro F1-Score
  - Confusion Matrix Heatmap

---

### E. Streamlit Dashboard Guide

1. **📊 Fleet Overview**: High-level KPIs (Total Drivers, Total Trips, Fleet Avg Score, Total Unsafe Events) and score comparison bar charts.
2. **👤 Driver Analysis**: Detailed breakdown per driver, speed metrics, event counts by category, and academic penalty calculation explanation.
3. **🛣️ Trip Analytics**: Interactive Plotly timeseries charts displaying **Speed vs. Time** (with 100 km/h limit line) and **Acceleration vs. Time** (with harsh brake & rapid accel boundaries).
4. **⚠️ Behavior Analytics**: Fleet-wide distribution donut charts and event type proportions.
5. **🤖 Machine Learning**: Model evaluation metrics, confusion matrix, feature importance ranking, and an interactive **Live Risk Predictor** with real-time sliders.
6. **☁️ Cloud Integration**: Status monitor for Amazon S3, configuration template, and verified backup button.

---

## ☁️ Cloud Integration (Amazon S3)

The application functions **100% locally by default**. Cloud backup is an optional feature.

### AWS Setup Instructions

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your AWS credentials and bucket name:
   ```ini
   AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
   AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   AWS_DEFAULT_REGION=us-east-1
   S3_BUCKET_NAME=my-driver-analytics-telemetry-bucket
   ```
3. Restart the Streamlit dashboard or run `python cloud_storage.py`.

### AWS Pricing & Free Tier Notice
- **Free Tier Allowance**: 5 GB of standard S3 storage, 20,000 GET requests, and 2,000 PUT requests per month for 12 months.
- **Standard Rates**: Approximately **$0.023 per GB/month** (US East).
- **Required IAM Permissions**: `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`.
- ⚠️ *Always delete temporary test buckets or set lifecycle expiration rules to prevent unexpected AWS charges.*

---

## 🧪 Running Tests

A comprehensive unit and integration test suite is included in `test_platform.py`:
```bash
python test_platform.py
```
Expected output:
```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.22s

OK
```
Tests cover:
- Telemetry simulation structure & physical bounds
- Behavior detection logic & severity tiering
- Safety scoring mathematical bounds & explanations
- SQLite database CRUD & queries
- Machine learning model loading & inference
- Cloud storage graceful local fallback

---

## 🎤 Presentation & Demo Guide

When demonstrating this project for academic or technical evaluations:

1. **Walk Through the Architecture**: Explain how raw vehicular sensors feed into rule detection, scoring, SQLite, and the ML classifier.
2. **Demonstrate Telemetry Generation**: Click **"⚡ Generate New Data"** in the sidebar to prove the dashboard renders dynamic, stored SQLite data rather than hardcoded mock figures.
3. **Showcase Trip Dynamics**: Go to **Trip Analytics** and demonstrate how speed and acceleration dips correlate with detected harsh braking and rapid acceleration markers.
4. **Demonstrate Machine Learning**: Navigate to the **Machine Learning** tab, show the Confusion Matrix, and adjust the input sliders (e.g. increase Harsh Braking to 4 and Max Speed to 120 km/h) to show the model instantly classify the trip as `High Risk`.
5. **Explain Local-First Cloud Design**: Point out how the app works seamlessly offline, and show how Amazon S3 is cleanly decoupled via Boto3 and environment variables.

---

## ⚠️ Limitations & Disclosures

- **Synthetic Simulation**: All telemetry data in this project is synthetically generated using physical models. In commercial telematics, datasets are gathered from OBD-II/CAN-bus devices and calibrated across diverse weather, GPS jitter, and vehicle weight classes.
- **Educational Scoring Model**: The 0–100 safety score and penalty weights represent a project-defined heuristic designed for demonstration and coursework, not an actuarial insurance underwriting standard.
- **Security**: Real AWS credentials must never be committed to source code or shared in public repositories.
