# Adaptive Machine Learning for Household Electricity Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Node.js 22](https://img.shields.io/badge/Node.js-22.x-brightgreen.svg)](https://nodejs.org/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Express-61DAFB.svg)](https://react.dev/)
[![Webpack 5](https://img.shields.io/badge/Bundler-Webpack%205%20(No%20Vite)-blueviolet.svg)](https://webpack.js.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![pytest](https://img.shields.io/badge/Tests-23%20passed-brightgreen.svg)](https://pytest.org/)

An end-to-end research and engineering framework for next-hour household electricity consumption forecasting and adaptive anomaly detection on the **UCI Individual Household Electric Power Consumption dataset**.

The system pairs a single reproducible **XGBoost** forecaster (22 strictly leak-free lag, rolling, and cyclical features) with a past-only **Rolling Median + Median Absolute Deviation (MAD)** anomaly detector using an anomaly-exclusion update policy, and serves a **mobile-friendly React 18 + Node.js / Express web dashboard** bundled with **Webpack 5** (zero Streamlit, zero Vite).

---

## Key Features

1. **Next-Hour Consumption Forecasting (`XGBRegressor`)**
   - Single reproducible XGBoost model with validation early stopping (`random_state=42`).
   - 22 strictly leak-free features: autoregressive lags (`1h, 2h, 3h, 24h, 48h, 168h`), past-only lagged rolling means/stds (`3h, 6h, 24h, 168h`), and calendar/cyclical encodings.
   - Strict chronological splitting (70% Train, 15% Validation, 15% Test) with zero temporal overlap.
   - Recursive multi-step horizon simulation (`12h`, `24h`, `48h`) via `FeatureEngineer.forecast_recursive_24h`.

2. **Adaptive Rolling Median + MAD Anomaly Detection**
   - Residual scoring: `s_t = |y_t - y_hat_t|`.
   - Past-only dynamic threshold over a 168-hour rolling buffer:
     `Threshold_t = Median(H_t) + k * 1.4826 * max(MAD(H_t), eps)`
   - Zero-MAD numerical safeguard (`eps = 1e-4`).
   - Contamination prevention (`exclude_anomalies=True`) so detected spikes do not inflate future thresholds.
   - Benchmarked against a validation-calibrated `FixedThresholdBaseline` (3-sigma static threshold).

3. **Controlled Synthetic Benchmark Suite**
   - Evaluates **Temporary Spikes**, **Sustained Shifts**, **Repeated Spikes**, and **Sudden Drops** with event-level detection rate, detection delay, and per-scenario recall.

4. **Mobile-Friendly React 18 + Express Dashboard (`http://localhost:5000`)**
   - **Responsive Navigation:** Sticky mobile header + fixed bottom tab bar on phones (`< 768px`), persistent sidebar on desktop (`>= 768px`).
   - **Overview:** Test MAE/RMSE/R², event capture rate, flagged hours, recent test stream chart, and verification checklist.
   - **Dataset Explorer:** Hourly dataset statistics, 24-hour diurnal load profile, weekly consumption profile, and validation audit.
   - **Forecast & Horizon Simulator:** Chronological split metrics and live interactive `12h / 24h / 48h` recursive rollout powered by the Python bridge (`src/api_bridge.py`).
   - **Anomaly Studio:** Interactive two-panel charts (consumption + anomalies, residual vs. MAD threshold), category filters, mobile event cards / desktop table, and CSV export.
   - **Benchmarks & Reports:** Adaptive MAD vs. Fixed Baseline comparison and one-click downloads for `.docx`, `.md`, `.json`, and `.csv` reports.

5. **Automated Test Suite (`pytest`)**
   - 23 unit and integration tests covering raw `?` parsing, hourly aggregation, chronological split integrity, zero future target leakage, model persistence, past-only MAD thresholds, zero-MAD safeguards, contamination prevention, and end-to-end execution.

---

## Project Structure

```text
mlproject/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── hourly_electricity_consumption.csv   # Processed hourly UCI dataset (34,589 rows)
├── run_pipeline.py                      # End-to-end CLI pipeline runner
├── validate_data.py                     # Schema, range, and split integrity validator
├── generate_accuracy_report.py          # Generates accuracy_report.json & .md
├── generate_docx_report.py              # Generates formatted Word (.docx) research report
├── src/
│   ├── __init__.py
│   ├── config.py                        # Dataclass configs & directory paths
│   ├── data_loader.py                   # UCI downloader, parser, hourly aggregator
│   ├── preprocessing.py                 # Cleaning, imputation & chronological split
│   ├── feature_engineering.py           # 22 leak-free features & recursive rollout
│   ├── model_training.py                # XGBRegressor training, evaluation & persistence
│   ├── anomaly_detector.py              # Adaptive MAD detector & Fixed 3-sigma baseline
│   ├── evaluation.py                    # Synthetic anomaly injector & Evaluator
│   ├── pipeline.py                      # End-to-end pipeline orchestrator
│   ├── visualization.py                 # Plotly chart helpers
│   └── api_bridge.py                    # JSON CLI bridge for Express backend
├── frontend/
│   ├── package.json
│   ├── webpack.config.js                # Webpack 5 production/dev config (no Vite)
│   ├── .babelrc
│   ├── server.js                        # Node.js Express server & REST API (port 5000)
│   ├── public/index.html
│   └── src/
│       ├── index.js
│       ├── App.jsx                      # 5-tab mobile-first React 18 dashboard
│       ├── components/Charts.jsx        # Touch & hover interactive SVG charts
│       └── styles/global.css            # Mobile-first responsive CSS
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   ├── test_anomaly_detector.py
│   ├── test_evaluation.py
│   └── test_integration.py
├── models/                              # Saved XGBoost model (.joblib) & metadata (.json)
└── reports/                             # Generated JSON, CSV, MD, and DOCX reports
```

---

## Quickstart & Installation

### 1. Python Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Tests
```powershell
python -m pytest
```
All 23 unit and integration tests pass.

### 3. Run Validation, ML Pipeline & Reports
```powershell
python validate_data.py
python run_pipeline.py
python generate_accuracy_report.py
python generate_docx_report.py
```

Observed Test Partition Performance:
- **Test MAE:** `0.3122 kWh`
- **Test RMSE:** `0.4560 kWh`
- **Test R²:** `0.5787`
- **Controlled Benchmark Event Detection Rate:** `100.0%` (`0.11h` average delay)

### 4. Build & Launch the React 18 + Express Web App
```powershell
cd frontend
npm install
npm run build
npm start
```
Open **http://localhost:5000** on desktop or mobile.
