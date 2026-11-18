"""End-to-end anomaly detection and benchmarking pipeline orchestrator."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.anomaly_detector import AdaptiveAnomalyDetector, FixedThresholdBaseline
from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    AnomalyDetectorConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    SyntheticScenarioConfig,
)
from src.data_loader import DataLoader
from src.evaluation import Evaluator, SyntheticAnomalyInjector
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    """Orchestrates data ingestion, training, anomaly detection, and evaluation."""

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        model_config: Optional[ModelConfig] = None,
        detector_config: Optional[AnomalyDetectorConfig] = None,
        scenario_config: Optional[SyntheticScenarioConfig] = None,
    ):
        self.data_config = data_config or DataConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.detector_config = detector_config or AnomalyDetectorConfig()
        self.scenario_config = scenario_config or SyntheticScenarioConfig()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def run_pipeline(
        self,
        force_recompute_data: bool = False,
        force_retrain_model: bool = False,
    ) -> Dict[str, Any]:
        """Execute the full end-to-end pipeline and persist reports."""
        loader = DataLoader(self.data_config)
        df_hourly = loader.process_and_save(force=force_recompute_data)

        prep = Preprocessor(self.data_config)
        df_clean = prep.clean_hourly_data(df_hourly)
        df_train, df_val, df_test = prep.split_chronological(df_clean)

        engineer = FeatureEngineer(self.feature_config, self.data_config)
        df_feats = engineer.create_features(df_clean)

        dt_col = self.data_config.datetime_col
        target_col = self.data_config.target_col
        train_max_dt = df_train[dt_col].max()
        val_max_dt = df_val[dt_col].max()

        feats_train = df_feats[df_feats[dt_col] <= train_max_dt]
        feats_val = df_feats[(df_feats[dt_col] > train_max_dt) & (df_feats[dt_col] <= val_max_dt)]
        feats_test = df_feats[df_feats[dt_col] > val_max_dt]

        X_train, y_train, _ = engineer.get_feature_matrix(feats_train)
        X_val, y_val, _ = engineer.get_feature_matrix(feats_val)
        X_test, y_test, ts_test = engineer.get_feature_matrix(feats_test)

        trainer = ModelTrainer(self.model_config)
        model_path = MODELS_DIR / "xgboost_forecaster.joblib"
        if model_path.exists() and not force_retrain_model:
            logger.info("Loading existing model from %s", model_path)
            trainer.load_model("xgboost_forecaster")
        else:
            logger.info("Training XGBoost forecaster...")
            trainer.train(X_train, y_train, X_val, y_val)
            trainer.save_model("xgboost_forecaster")

        forecasting_metrics = trainer.evaluate(X_test, y_test)
        preds_val = trainer.predict(X_val)
        preds_test = trainer.predict(X_test)

        # Calibrate fixed baseline on validation residuals
        val_residuals = y_val.to_numpy(dtype=float) - preds_val
        fixed_baseline = FixedThresholdBaseline()
        calibrated_thresh = fixed_baseline.fit_from_residuals(val_residuals, multiplier=3.0)

        # Run Adaptive Detector on the real test stream and export CSV
        detector = AdaptiveAnomalyDetector(self.detector_config)
        real_test_anomalies = detector.detect(
            actual=y_test.to_numpy(dtype=float),
            predicted=preds_test,
            timestamps=ts_test,
        )
        anomalies_csv_path = REPORTS_DIR / "detected_anomalies.csv"
        real_test_anomalies.to_csv(anomalies_csv_path, index=False)
        logger.info("Saved detected anomalies log to %s", anomalies_csv_path)

        # Controlled synthetic benchmark evaluation
        test_df = pd.DataFrame({dt_col: ts_test.values, target_col: y_test.values})
        injector = SyntheticAnomalyInjector(self.scenario_config)
        df_injected, events_df = injector.inject_scenarios(test_df, target_col)

        adaptive_bench = detector.detect(
            actual=df_injected[target_col],
            predicted=preds_test,
            timestamps=ts_test,
        )
        adaptive_bench["ground_truth_label"] = df_injected["ground_truth_label"].values
        adaptive_bench["ground_truth_scenario"] = df_injected["ground_truth_scenario"].values

        baseline_bench = fixed_baseline.detect(
            actual=df_injected[target_col],
            predicted=preds_test,
            timestamps=ts_test,
        )
        baseline_bench["ground_truth_label"] = df_injected["ground_truth_label"].values
        baseline_bench["ground_truth_scenario"] = df_injected["ground_truth_scenario"].values

        adaptive_eval = Evaluator.evaluate_scenarios(adaptive_bench, events_df)
        baseline_eval = Evaluator.evaluate_scenarios(baseline_bench, events_df)
        baseline_eval["calibrated_threshold_kwh"] = round(calibrated_thresh, 4)

        comparison_report = {
            "generated_at": datetime.now().isoformat(),
            "forecasting_metrics": forecasting_metrics,
            "adaptive_detector": adaptive_eval,
            "fixed_baseline": baseline_eval,
        }

        report_path = REPORTS_DIR / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2)
        logger.info("Saved evaluation report to %s", report_path)

        return {
            "comparison_report": comparison_report,
            "adaptive_results": real_test_anomalies,
        }
