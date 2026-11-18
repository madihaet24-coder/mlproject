# Adaptive Machine Learning for Household Electricity Anomaly Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Node.js 20+](https://img.shields.io/badge/Node.js-24.x-brightgreen.svg)](https://nodejs.org/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Node.js-61DAFB.svg)](https://react.dev/)
[![Webpack 5](https://img.shields.io/badge/Bundler-Webpack%205%20(No%20Vite)-blueviolet.svg)](https://webpack.js.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![pytest](https://img.shields.io/badge/Tests-23%20passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic research project implementing end-to-end forecasting and adaptive anomaly detection on the official **UCI Individual Household Electric Power Consumption dataset**.

The framework predicts expected next-hour electricity consumption using **XGBoost** with strictly lagged features, detects anomalous behavior using historical residual **rolling Median and Median Absolute Deviation (MAD)** thresholds, prevents baseline contamination through anomaly exclusion policies, and provides a modern **React 18 + Node.js / Express web dashboard (powered by Webpack 5, strictly zero Vite)**.

---

## 🌟 Key Features

1. **Next-Hour Consumption Forecasting:**
   - Single reproducible `XGBRegressor` forecasting model with validation early-stopping.
   - Strictly leak-free autoregressive lags ($t-1, t-2, t-24, t-168$) and lagged rolling statistics.
   - Preserves strict chronological monotonic ordering (70% Train, 15% Validation, 15% Test).

2. **Adaptive Anomaly Detection:**
   - Residual-based scoring: $s_t = |y_t - \hat{y}_t|$.
   - Time-varying dynamic threshold: $\text{Threshold}_t = \text{Median}(H_t) + k \times 1.4826 \times \max(\text{MAD}(H_t), \epsilon)$.
   - Strict past-only calculation (no future leakage).
   - Numerical safeguards against zero-MAD collapse.
   - Contamination prevention update policy (`exclude_anomalies`) ensuring abnormal spikes do not artificially inflate future detection thresholds.

3. **Controlled Benchmark Suite:**
   - Five reproducible controlled synthetic evaluation scenarios:
     1. *Temporary Spikes* (1-2 hours)
     2. *Sustained Level Shifts* (36 hours)
     3. *Repeated Spikes* (periodic bursts)
     4. *Sudden Drops* (near-zero usage)
     5. *Normal Routine Changes* (long-term lifestyle change; validates adaptive threshold convergence)
   - Benchmarked directly against a calibrated *Fixed-Threshold Baseline*.

4. **Interactive Research Dashboard:**
   - **Overview Dashboard:** High-level KPIs, system status, and actual vs. forecast charts.
   - **Dataset Explorer:** Date filters, missing data analysis, diurnal and weekly consumption profiles, and raw/processed CSV exports.
   - **Model Training:** Split summaries, hyperparameters, training logs, and validation metrics ($MAE, RMSE, R^2$).
   - **Anomaly Detection:** Two-panel interactive Plotly view (consumption + residual threshold), anomaly filters, and CSV export.
   - **Evaluation & Comparison:** Detailed benchmark tables, precision, recall, false alarms per day, and error analysis.
   - **Project Documentation:** Complete methodology, formulas, citations, and limitations.

5. **Production Testing Suite:**
   - 23 comprehensive unit and integration tests covering dataset parsing, leakage prevention, split integrity, persistence, numerical edge cases, and reproducibility.

---

## 📁 Project Structure

```text
ML EL/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── run_pipeline.py            # CLI pipeline runner
├── app.py                    # Streamlit interactive web dashboard
├── data/
│   ├── raw/                  # Preserved downloaded raw UCI files
│   └── processed/            # Aggregated hourly datasets (CSV/Parquet)
├── models/
│   ├── xgboost_forecaster.joblib           # Trained model artifact
│   └── xgboost_forecaster_metadata.json    # Features and hyperparameters
├── reports/
│   ├── evaluation_report.json              # Evaluation benchmark results
│   ├── detected_anomalies.csv              # Exported anomaly log
│   └── research_report.md                  # Comprehensive academic report
├── src/
│   ├── __init__.py
│   ├── config.py             # Dataclass configurations & hyperparameters
│   ├── data_loader.py        # Programmatic UCI dataset downloader & parser
│   ├── preprocessing.py      # Missing-value policies & chronological split
│   ├── feature_engineering.py# Lag & rolling features (leak-free)
│   ├── model_training.py     # XGBoost training & metric calculation
│   ├── anomaly_detector.py   # Adaptive MAD detector & baseline
│   ├── evaluation.py         # Synthetic injection & evaluation metrics
│   ├── pipeline.py           # End-to-end pipeline orchestrator
│   └── visualization.py      # Clean Plotly visualization routines
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest synthetic fixtures
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_model_training.py
    ├── test_anomaly_detector.py
    ├── test_evaluation.py
    └── test_integration.py   # End-to-end pipeline validation
```

---

## ⚙️ System Requirements

- **Operating System:** Windows, macOS, or Linux.
- **Python Version:** Python 3.11 or compatible (3.10 - 3.12).
- **RAM:** Minimum 4 GB (8 GB recommended for raw minute processing).
- **Disk Space:** ~500 MB for dataset and virtual environment.

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Workspace
```powershell
cd "c:\Users\ADMIN\Desktop\ML EL"
```

### 2. Create and Activate Virtual Environment
```powershell
# Create venv
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
# source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Running Automated Tests

Run the full pytest suite (all 23 unit & integration tests):
```powershell
python -m pytest
```

Output:
```text
======================= 23 passed in 2.08s =======================
```

The tests strictly validate:
- Missing-value handling and `?` symbol parsing.
- Unit conversion (minute kW to hourly kWh).
- Monotonic chronological splitting with zero overlapping timestamps.
- Programmatic verification of **no future target leakage** across all features.
- Model artifact persistence and reloading consistency.
- Historical threshold calculation without future residuals.
- Numerical safeguards for zero or near-zero MAD.
- Controlled synthetic anomaly injection and reproducibility.

---

## 🏃 Running the Full ML Pipeline

Execute the end-to-end pipeline from the command line:
```powershell
python run_pipeline.py
```

This will automatically:
1. Download the official UCI dataset from the repository (if not already local).
2. Clean and aggregate minute readings into 34,589 hourly records.
3. Compute chronological train/val/test splits.
4. Extract 22 leak-free lag and temporal features.
5. Train the XGBoost regressor with early stopping.
6. Calibrate the baseline threshold on validation data.
7. Run the Adaptive MAD detector and evaluate against controlled test scenarios.
8. Save results to `reports/evaluation_report.json` and `reports/detected_anomalies.csv`.

---

## 🌐 Launching the React 18 + Node.js Web Application

The project features a **React 18 + Node.js/Express web dashboard** built with **Webpack 5 (strictly zero Vite)**:

### 1. Build and Run the React + Express Dashboard
```powershell
# Navigate to frontend directory
cd frontend

# Run production build (Webpack 5)
npm run build

# Start the Node.js Express server
npm start
```

Open your browser to: **http://localhost:5000**

### 2. Available Frontend Features:
- **System Overview & KPIs:** Live system health, test MAE/RMSE/R², detection rates, and pipeline status.
- **Data Explorer:** Diurnal 24h consumption profiles, weekly patterns, missing data distribution, and dataset statistics.
- **24-Hour Horizon Forecaster:** Interactive residual decomposition, actual vs. forecast curves, and next-24h simulation.
- **Anomaly Detection Studio:** Dynamic rolling MAD threshold curves, anomaly filter tables, and event inspection.
- **Evaluation & Benchmarks:** Side-by-side comparison between Adaptive MAD vs. Fixed-Threshold Baseline across all 5 synthetic scenarios.
- **Research Documentation & Downloads:** One-click downloads for `.docx` Research Report, `.csv` detected anomalies, and JSON accuracy reports.

---

## 📊 Evaluation Results Summary

Results evaluated on the untouched final 15% chronological test set (4,733 hourly observations):

### Forecasting Model Performance (XGBoost)
- **Test MAE:** `0.3214 kWh`
- **Test RMSE:** `0.4618 kWh`
- **Test R²:** `0.5740`

### Anomaly Detection Comparison

| Evaluation Metric | Adaptive MAD Detector | Fixed-Threshold Baseline | Advantage |
| :--- | :---: | :---: | :---: |
| **Event-Level Detection Rate** | **100.0% (12/12)** | 91.7% (11/12) | **Complete event capture** |
| **Average Detection Delay** | **0.00 hours** | 0.00 hours | Real-time response |
| **Injected Anomaly Recall** | **96.25% (77/80)** | 70.00% (56/80) | **+26.25% higher sensitivity** |
| **Controlled Window Precision** | **24.9%** | 55.4% | Robust to local drift |
| **Baseline Adaptability** | **Dynamic Rolling Window** | Static ($1.516$ kWh) | Avoids permanent false alarms |

---

## 📖 Citation

If using this project or the dataset in academic work:

```bibtex
@misc{hebrail2012household,
  author       = {Hebrail, Georges and Berard, Alice},
  title        = {Individual Household Electric Power Consumption},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C58888}
}
```

---

## ⚖️ Known Limitations

1. **Unannotated Real-World Spikes:** The UCI dataset lacks occupant activity logs. Real appliance activations (such as heating or cooking) naturally create residuals that the detector flags as anomalous, which are treated as "false alarms" when evaluating against purely synthetic labels.
2. **Weather Covariates:** Incorporating external ambient temperature would further reduce forecasting error during seasonal extremes.

