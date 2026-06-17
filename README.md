# 🏎️ F1 Tire Crossover & Degradation Predictor

> ⚠️ **Status: Active Development (Work in Progress)** > The data engineering, extraction, and time-series alignment pipelines are complete. The core Machine Learning modeling phase (XGBoost/Random Forest) is currently under active development.

## Overview
**`f1-crossover-predict`** is a production-ready data engineering and machine learning pipeline designed to model Formula 1 race strategy. Utilizing the `fastf1` API, this repository extracts, cleans, and merges high-resolution lap telemetry with asynchronous trackside weather conditions. 

The core objective of this project is to predict tire life drop-off and classify the statistically optimal lap for compound crossovers (e.g., Intermediates to Slicks) using gradient-boosted decision trees, strictly engineered to prevent time-series data leakage.

## 🏗️ Architecture & Pipeline (Current State)

### Phase 1 & 2: Extraction & Asynchronous Cleaning (Completed)
Real-world motorsport data is highly asynchronous; track temperature and weather sensors ping at different frequencies than the timing beam. The data pipeline handles this by:
* **Asynchronous Alignment:** Forward-filling and aligning weather telemetry precisely to lap timestamps.
* **Physics-First Filtering:** Dropping categorical identifiers (like `Driver`) to force the model to learn pure aerodynamic and thermal physics (downforce degradation, track temperature deltas) rather than overfitting to specific driver profiles. Null compound anomalies are strictly amputated to preserve wet-weather physics integrity.

### Data Dimensions
The current master dataset consists of **~13,700 rows** of purely accurate racing laps (excluding safety cars, out-laps, and in-laps) curated from a strategic mix of 15 high-variance races (e.g., Zandvoort extreme wet-to-dry, Bahrain high thermal degradation).

## 🚀 Roadmap

- [x] **Phase 1:** API and historical session extraction.
- [x] **Phase 2:** Asynchronous time-series cleaning and dataset compilation.
- [ ] **Phase 3:** Feature Engineering (Delta temps, rolling wetness metrics).
- [ ] **Phase 4:** Baseline Model Training (Random Forest Regressor for LapTime, XGBoost for Crossover Classification) utilizing strict chronological splitting.
- [ ] **Phase 5:** Model Evaluation & Hyperparameter Tuning.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Engineering:** Pandas, NumPy, Matplotlib, Seaborn, FastF1 API
* **Machine Learning:** Scikit-Learn
* **Environment:** python-dotenv (Path isolation)

## 💻 Local Setup (For Contributors/Reviewers)
1. Clone the repository.
2. Install dependencies: `pip install fastf1 pandas numpy matplotlib seaborn scikit-learn python-dotenv jupyter`
3. Create a `.env` file at the root and define your absolute path for data storage:
   ```text
   DATA_PATH="C:/your/absolute/path/data/processed"
