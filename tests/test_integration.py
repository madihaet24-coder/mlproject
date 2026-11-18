"""Integration tests for end-to-end pipeline, validation, and visualization."""

import plotly.graph_objects as go
from src.anomaly_detector import AdaptiveAnomalyDetector
from src.config import AnomalyDetectorConfig, SyntheticScenarioConfig
from src.evaluation import Evaluator, SyntheticAnomalyInjector
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.preprocessing import Preprocessor
from src.visualization import (
    plot_actual_vs_predicted,
    plot_anomaly_detection_dashboard,
    plot_comparison_metrics,
    plot_consumption_patterns,
)


def test_end_to_end_synthetic_workflow(synthetic_hourly_df):
    prep = Preprocessor()
    df_clean = prep.clean_hourly_data(synthetic_hourly_df)
    df_train, df_val, df_test = prep.split_chronological(df_clean)

    engineer = FeatureEngineer()
    df_feats = engineer.create_features(df_clean)

    train_max = df_train["Datetime"].max()
    val_max = df_val["Datetime"].max()
    f_train = df_feats[df_feats["Datetime"] <= train_max]
    f_val = df_feats[(df_feats["Datetime"] > train_max) & (df_feats["Datetime"] <= val_max)]
    f_test = df_feats[df_feats["Datetime"] > val_max]

    X_train, y_train, _ = engineer.get_feature_matrix(f_train)
    X_val, y_val, _ = engineer.get_feature_matrix(f_val)
    X_test, y_test, ts_test = engineer.get_feature_matrix(f_test)

    trainer = ModelTrainer()
    trainer.train(X_train, y_train, X_val, y_val)
    preds = trainer.predict(X_test)

    det = AdaptiveAnomalyDetector(AnomalyDetectorConfig(window_size=48, min_history=12))
    res = det.detect(y_test.values, preds, ts_test)
    assert len(res) == len(y_test)


def test_visualization_figures_render(synthetic_hourly_df):
    det = AdaptiveAnomalyDetector(AnomalyDetectorConfig(window_size=48, min_history=12))
    res = det.detect(
        actual=synthetic_hourly_df["Global_active_power"],
        predicted=synthetic_hourly_df["Global_active_power"] * 0.98,
        timestamps=synthetic_hourly_df["Datetime"],
    )
    assert isinstance(plot_actual_vs_predicted(res), go.Figure)
    assert isinstance(plot_anomaly_detection_dashboard(res), go.Figure)
    assert isinstance(plot_consumption_patterns(synthetic_hourly_df), go.Figure)
    assert isinstance(
        plot_comparison_metrics(
            {
                "Adaptive": {"precision": 0.9, "recall": 0.95, "f1_score": 0.92},
                "Baseline": {"precision": 0.7, "recall": 0.75, "f1_score": 0.72},
            }
        ),
        go.Figure,
    )


def test_data_validator_on_real_dataset():
    from validate_data import DataValidator

    validator = DataValidator()
    res = validator.validate_processed_data()
    assert res["overall_status"] == "PASSED"
    assert res["chronological_split_integrity"]["zero_temporal_leakage_verified"] is True
