# Adaptive Machine Learning for Household Electricity Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Node.js 22](https://img.shields.io/badge/Node.js-22.x-brightgreen.svg)](https://nodejs.org/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Express-61DAFB.svg)](https://react.dev/)
[![Webpack 5](https://img.shields.io/badge/Bundler-Webpack%205%20(No%20Vite)-blueviolet.svg)](https://webpack.js.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![pytest](https://img.shields.io/badge/Tests-29%20passed-brightgreen.svg)](https://pytest.org/)

An end-to-end research and engineering framework for next-hour household electricity consumption forecasting and selective adaptive anomaly detection on the **UCI Individual Household Electric Power Consumption dataset**.

The system pairs a single reproducible **XGBoost** forecaster (22 strictly leak-free lag, rolling, and cyclical features) with a past-only **Rolling Median + Median Absolute Deviation (MAD)** anomaly detector featuring validation-tuned sensitivity ($k = 4.0$), minimum error floor, two-tier spike persistence, consecutive anomaly event grouping, and controlled re-baselining — served through a **mobile-friendly React 18 + Node.js / Express web dashboard** bundled with **Webpack 5** (zero Streamlit, zero Vite).

---

## Key Features

1. **Next-Hour Consumption Forecasting (`XGBRegressor`)**
   - Single reproducible XGBoost model with validation early stopping (`random_state=42`).
   - 22 strictly leak-free features: autoregressive lags (`1h, 2h, 3h, 24h, 48h, 168h`), past-only lagged rolling means/stds (`3h, 6h, 24h, 168h`), and calendar/cyclical encodings.
   - Strict chronological splitting (70% Train, 15% Validation, 15% Test) with zero temporal overlap.
   - Recursive multi-step horizon simulation (`12h`, `24h`, `48h`) via `FeatureEngineer.forecast_recursive_24h`.

2. **Selective Adaptive Rolling Median + MAD Anomaly Detection**
   - **Validation-Selected Sensitivity ($k = 4.0$):** Sweeps candidate values $k \in \{3.0, 3.5, 4.0, 4.5\}$ on chronological validation data and selects $k = 4.0$ to suppress routine noise without missing genuine anomalies.
   - **Minimum Error Condition & Drop Preservation:** Enforces `min_residual_kwh = 0.65` kWh against minor fluctuations while using an asymmetric drop safeguard (`<= 0.18 * predicted`) so genuine sudden power drops remain detectable.
   - **Two-Tier Persistence & Event Grouping:** Requires `persistence = 2` consecutive hours for moderate deviations, triggers immediately (1 hour) for severe single-hour spikes ($\ge 1.35 \times T_t$) or near-total drops, and groups consecutive flagged hours into single discrete anomaly events (`detected_anomaly_events.csv`).
   - **Baseline Protection & Controlled Re-baselining:** Strictly past-only rolling 168-hour buffer excludes flagged anomalies from contaminating the normal baseline, preserves unflagged variance, and re-baselines cleanly after `rebaseline_hours = 48` of a sustained new consumption routine.

3. **Before vs. After False-Alarm Reduction**
   - **Validation Partition (Real Unlabelled Stream):** Flagged hours reduced from `607` (`11.70%`) to `73` (`1.41%`, **-88.0%**); grouped events reduced from `475` to `63` (**-86.7%**).
   - **Test Partition (Real Unlabelled Stream):** Flagged hours reduced from `652` (`12.57%`) to `76` (`1.46%`, **-88.3%**); grouped events reduced from `448` to `59` (**-86.8%**).
   - **Controlled Synthetic Test Benchmark:** `100.0%` event-level detection rate (`0.0h` average delay), `100.0%` recall across all 4 injected scenarios (`temporary_spike`, `repeated_spikes`, `sustained_shift`, `sudden_drop`), `89.9%` controlled precision, and `94.7%` controlled F1.
   - *Note:* Alerts identify potential unusual electricity consumption anomalies rather than confirmed physical electricity leakage; real-world UCI hours are unlabelled, while Precision/Recall/F1 are computed on controlled synthetic benchmarks.

4. **Mobile-Friendly React 18 + Express Dashboard (`http://localhost:5000`)**
   - **Responsive Navigation:** Sticky mobile header + fixed bottom tab bar on phones (`< 768px`), persistent sidebar on desktop (`>= 768px`).
   - **Overview:** Test MAE/RMSE/R², grouped events, flagged-hour reduction vs. legacy `k=3.0`, detector parameter summary, and Before-vs-After comparison table.
   - **Dataset Explorer:** Hourly dataset statistics, 24-hour diurnal load profile, weekly consumption profile, and validation audit.
   - **Forecast & Horizon Simulator:** Chronological split metrics and live interactive `12h / 24h / 48h` recursive rollout powered by `src/api_bridge.py`.
   - **Anomaly Studio:** Live sensitivity selector (`k = 3.0, 3.5, 4.0, 4.5`), interactive two-panel charts (consumption + anomalies, residual vs. MAD threshold), grouped events vs. hourly view toggle, and CSV exports.
   - **Benchmarks & Reports:** Validation $k$-sensitivity sweep table, Before-vs-After validation/test table, Adaptive MAD vs. Fixed Baseline comparison, and `.docx`/`.md`/`.json`/`.csv` downloads.

5. **Automated Test Suite (`pytest`)**
   - 29 unit and integration tests covering `?` parsing, hourly aggregation, chronological split integrity, zero future target leakage, model persistence, past-only MAD thresholds, $k$-sensitivity monotonicity, minimum residual floor, severe single-hour spike override, consecutive event grouping, and controlled re-baselining.

---

## Project Structure

```text
mlproject/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── run_project.bat                      # Windows one-click launcher
├── hourly_electricity_consumption.csv   # Processed hourly UCI dataset (34,589 rows)
├── run_pipeline.py                      # End-to-end CLI pipeline runner
├── validate_data.py                     # Schema, range, and split integrity validator
├── generate_accuracy_report.py          # Generates accuracy_report.json & .md
├── generate_docx_report.py              # Generates formatted Word (.docx) research report
├── src/
│   ├── __init__.py
│   ├── config.py                        # Dataclass configs & legacy comparison preset
│   ├── data_loader.py                   # UCI downloader, parser, hourly aggregator
│   ├── preprocessing.py                 # Cleaning, imputation & chronological split
│   ├── feature_engineering.py           # 22 leak-free features & recursive rollout
│   ├── model_training.py                # XGBRegressor training, evaluation & persistence
│   ├── anomaly_detector.py              # Adaptive MAD detector, event grouping & baseline
│   ├── evaluation.py                    # Synthetic injector, k-sweep & before/after evaluator
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
All 29 unit and integration tests pass.

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
- **Controlled Benchmark Event Detection Rate:** `100.0%` (`0.0h` average delay)

### 4. Build & Launch the React 18 + Express Web App
```powershell
cd frontend
npm install
npm run build
npm start
```
Or run [run_project.bat](run_project.bat) on Windows, then open **http://localhost:5000** on desktop or mobile.
