# 🏎️ F1 Tire Crossover & Degradation Predictor

> ⚠️ **Status: Active Development (Work in Progress)** 
> The data engineering, extraction, time-series alignment, and target normalization pipelines are complete. The pace degradation regression engine has completed feature engineering and time-series hyperparameter tuning. The project is currently transitioning into compound crossover classification and neural network benchmarking.

## Overview
**`f1-crossover-predict`** is a production-ready data engineering and machine learning pipeline designed to model Formula 1 race strategy. Utilizing the `fastf1` API, this repository extracts, cleans, and merges high-resolution lap telemetry with asynchronous trackside weather conditions. 

The core objective of this project is to predict tire life drop-off and classify the statistically optimal lap for compound crossovers (e.g., Intermediates to Slicks) using gradient-boosted decision trees, strictly engineered to prevent time-series data leakage.

## 🏗️ Architecture & Pipeline

### Phase 1 & 2: Extraction & Asynchronous Cleaning 
Real-world motorsport data is highly asynchronous; track temperature and weather sensors ping at different frequencies than the timing beam. The data pipeline handles this by:
* **Asynchronous Alignment:** Forward-filling and aligning weather telemetry precisely to lap timestamps.
* **Physics-First Filtering:** Dropping categorical identifiers (like `Driver`) to force the model to learn pure aerodynamic and thermal physics (downforce degradation, track temperature deltas) rather than overfitting to specific driver profiles. Null compound anomalies are strictly amputated to preserve wet-weather physics integrity.

### Phase 3: Handling Track Variance via Target Normalization
To prevent the model from memorizing track lengths (e.g., a 72-second lap at Monaco vs. a 105-second lap at Spa), the concept of absolute track length was completely removed. 
* **Target Attribute Shift:** Instead of predicting the raw `LapTime_s`, the pipeline groups data by race event and calculates the `time_delta` (the pace drop-off compared to the absolute fastest lap of that specific session).
* **Pure Physics Engine:** This ensures the algorithm purely learns the physical penalty of tire degradation (e.g., +3.0s per lap of age) regardless of the track being driven.

### Phase 4: Time-Series Feature Engineering
To capture non-linear wear characteristics and cumulative atmospheric dynamics, several domain-driven features were engineered:
* **Tire Cliff Modeling (`TyreLife_Sq`):** Captures the exponential degradation of grip when rubber wears to the carcass rather than forcing an artificial linear relationship.
* **Cumulative Wetness (`Rainfall_Rolling4`):** A 4-lap rolling sum of the rain sensor, accounting for standing water and track saturation after rainfall ceases.
* **Thermal Inertia (`TrackTemp_Rolling4`):** A 4-lap rolling average capturing the thermal memory of the asphalt rather than single-lap sensor noise.

## 📊 Exploratory Data Analysis (EDA) Insights
Visualizing the ~13,700 rows of curated racing laps revealed critical motorsport realities that dictated the modeling approach:
* **The Wet Weather Distortion:** Pearson correlation initially showed a negative relationship between `TyreLife` and pace. This was driven by chaotic wet races where laps on fresh tires (low `TyreLife`) were naturally 30–70 seconds slower than dry laps, temporarily skewing the statistical baseline.
* **Temperature Collinearity & Dominance:** `Rainfall` proved to be the dominant feature (0.52 correlation with slower pace). Furthermore, `AirTemp` and `TrackTemp` exhibited heavy collinearity (0.86), with all massive time deltas (+20s) isolated strictly to track temperatures below 35°C.
* **Compound Encoding:** Categorical tire compounds were one-hot encoded to prevent standard integer mapping (0–4) from implying an artificial mathematical ranking.

## 🤖 Model Evolution & Performance Benchmarks
To prevent time-series data leakage, models avoid standard random shuffling and strictly enforce an outer **80/20 chronological holdout split**. 

Hyperparameter search was executed using inner **`TimeSeriesSplit` (tscv)** cross-validation to preserve temporal structure during candidate selection.

| Pipeline Iteration | Architecture & Strategy | RMSE | MAE |
|---|---|---|---|
| **Phase 4 Baseline** | Raw XGBoost (`n_est=100, lr=0.1, depth=5`) | 4.3226s | 3.3942s |
| **Huber Loss Trial** | Pseudo-Huber Objective (Default params suppressed updates) | 55.9517s | 38.9129s |
| **Window=3 Rolling Features** | XGBoost + `TyreLife_Sq` + 3-lap rolling weather | 2.8939s | 2.4553s |
| **Window=4 Rolling Features** | XGBoost + `TyreLife_Sq` + 4-lap rolling weather | 2.7971s | 2.3730s |
| **Initial Parameter Tuning** | Manual LR tuning (`lr=0.05`) | 2.7313s | 2.3348s |
| **TimeSeriesSplit CV Tuning** | RandomizedSearch with temporal cross-validation | 2.6646s | 2.3099s |
| **Expanded Search Grid** | Constrained shallow-tree ensemble optimization | **2.6271s** | **2.2593s** |

### Optimized Regressor Configuration
```python
best_params = {
    'n_estimators': 600,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 7,
    'subsample': 0.65,
    'colsample_bytree': 0.55,
    'reg_lambda': 1.0,
    'reg_alpha': 0.01
}
```
🚀 Roadmap
[x] Phase 1: API and historical session extraction.
[x] Phase 2: Asynchronous time-series cleaning and dataset compilation.
[x] Phase 3: Feature Engineering (Delta temps, target normalization).
[x] Phase 4: Baseline Model Training (XGBoost Regressor for time delta) utilizing strict chronological splitting.
[x] Phase 5: Advanced Feature Engineering (Rolling temperature/wetness windows, non-linear tire cliff modeling).
[x] Phase 6: TimeSeriesSplit Cross-Validated Hyperparameter Optimization.
[ ] Phase 7: XGBoost Compound Crossover Classification.
[ ] Phase 8: Comparative Study (Gradient Boosted Trees vs. Multi-Layer Perceptron Neural Networks).

🛠️ Tech Stack
Language: Python 3.10+
Data Engineering: Pandas, NumPy, FastF1 API
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-Learn, XGBoost
Environment: python-dotenv, Jupyter (Exploratory Notebooks)

💻 Local Setup (For Contributors/Reviewers)
1. Clone the repository:
2. Install core dependencies:
```bash
   pip install fastf1 pandas numpy matplotlib seaborn scikit-learn xgboost python-dotenv jupyter
```
3. Create a .env file at the root and define your absolute paths for data storage:
```bash
   DATA_PATH="C:/your/absolute/path/data/processed"
   LOAD_PATH="C:/your/absolute/path/data/processed/data.csv"
   LOAD_CLEAN_PATH="C:/your/absolute/path/data/processed/cleaned.csv"
   LOAD_TRAIN_PATH="C:/your/absolute/path/data/processed/train.csv"
```