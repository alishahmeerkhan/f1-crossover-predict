# 🏎️ F1 Tire Crossover & Degradation Predictor

> ⚠️ **Status: Active Development (Work in Progress)** 
> The data engineering, extraction, time-series alignment, and target normalization pipelines are complete. The project has successfully established baseline Machine Learning models (XGBoost Regressor) and is moving into advanced feature engineering and crossover classification.

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

## 📊 Exploratory Data Analysis (EDA) Insights
Visualizing the ~13,700 rows of curated racing laps revealed critical motorsport realities that dictated the modeling approach:
* **The Wet Weather Distortion:** Pearson correlation initially showed a negative relationship between `TyreLife` and pace. This was driven by chaotic wet races where laps on fresh tires (low `TyreLife`) were naturally 30-70 seconds slower than dry laps, temporarily skewing the statistical baseline.
* **Temperature Collinearity & Dominance:** `Rainfall` proved to be the dominant feature (0.52 correlation with slower pace). Furthermore, `AirTemp` and `TrackTemp` exhibited heavy collinearity (0.86), with all massive time deltas (+20s) isolated strictly to track temperatures below 35°C.
* **Compound Encoding:** Categorical tire compounds were one-hot encoded to prevent standard integer mapping (0-4) from implying an artificial mathematical ranking.

## 🤖 Model Training & Baseline Results
To prevent time-series data leakage, the models avoid standard random shuffling and instead utilize a strict **80/20 chronological split**. This ensures the algorithm learns from historical laps to predict future laps, mimicking live pit-wall forecasting.

**Baseline Model:** `XGBRegressor` (n_estimators=100, learning_rate=0.1, max_depth=5)
* **Mean Absolute Error (MAE):** 3.39 seconds
* **Root Mean Squared Error (RMSE):** 4.32 seconds

*Note: The current RMSE indicates the model is generally accurate but is occasionally penalized heavily by the chaotic wet-to-dry crossover laps, establishing the need for cumulative rolling features.*

## 🚀 Roadmap

- [x] **Phase 1:** API and historical session extraction.
- [x] **Phase 2:** Asynchronous time-series cleaning and dataset compilation.
- [x] **Phase 3:** Feature Engineering (Delta temps, target normalization).
- [x] **Phase 4:** Baseline Model Training (XGBoost Regressor for time delta) utilizing strict chronological splitting.
- [ ] **Phase 5:** Advanced Feature Engineering (Rolling temperature averages, cumulative track wetness, non-linear tire cliff modeling).
- [ ] **Phase 6:** XGBoost Crossover Classification & Hyperparameter Tuning.
- [ ] **Phase 7:** Comparative Study (Gradient Boosted Trees vs. Multi-Layer Perceptron Neural Networks).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Engineering:** Pandas, NumPy, FastF1 API
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Environment:** python-dotenv, Jupyter (Exploratory Notebooks)

## 💻 Local Setup (For Contributors/Reviewers)
1. Clone the repository.
2. Install core dependencies: 
   ```bash
   pip install fastf1 pandas numpy matplotlib seaborn scikit-learn xgboost python-dotenv jupyter
3. Create a .env file at the root and define your absolute paths for data storage:

```bash
   DATA_PATH="C:/your/absolute/path/data/processed"
   LOAD_PATH="C:/your/absolute/path/data/processed/data.csv"
   LOAD_CLEAN_PATH="C:/your/absolute/path/processed/cleaned.csv"