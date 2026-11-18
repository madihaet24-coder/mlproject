"""Unit tests for src.evaluation."""

import numpy as np
import pandas as pd
from src.anomaly_detector import AdaptiveAnomalyDetector
from src.config import AnomalyDetectorConfig, SyntheticScenarioConfig
from src.evaluation import Evaluator, SyntheticAnomalyInjector


def test_synthetic_injection_reproducibility(synthetic_hourly_df):
    inj1 = SyntheticAnomalyInjector(SyntheticScenarioConfig(random_seed=42))
    inj2 = SyntheticAnomalyInjector(SyntheticScenarioConfig(random_seed=42))

    df1, ev1 = inj1.inject_scenarios(synthetic_hourly_df)
    df2, ev2 = inj2.inject_scenarios(synthetic_hourly_df)

    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(ev1, ev2)


def test_synthetic_injection_contains_all_scenarios(synthetic_hourly_df):
    inj = SyntheticAnomalyInjector(SyntheticScenarioConfig(random_seed=42))
    df_inj, events_df = inj.inject_scenarios(synthetic_hourly_df)

    expected_scenarios = {"temporary_spike", "sustained_shift", "repeated_spikes", "sudden_drop"}
    assert expected_scenarios.issubset(set(events_df["scenario"].unique()))
    assert df_inj["ground_truth_label"].sum() == events_df["duration_hours"].sum()


def test_evaluator_outputs_valid_metrics(synthetic_hourly_df):
    inj = SyntheticAnomalyInjector(SyntheticScenarioConfig(random_seed=42))
    df_inj, events_df = inj.inject_scenarios(synthetic_hourly_df)

    det = AdaptiveAnomalyDetector(AnomalyDetectorConfig(sensitivity=3.0, window_size=168))
    res = det.detect(
        actual=df_inj["Global_active_power"],
        predicted=synthetic_hourly_df["Global_active_power"],
        timestamps=df_inj["Datetime"],
    )
    res["ground_truth_label"] = df_inj["ground_truth_label"].values
    res["ground_truth_scenario"] = df_inj["ground_truth_scenario"].values

    eval_out = Evaluator.evaluate_scenarios(res, events_df)
    assert 0.0 <= eval_out["event_level"]["event_detection_rate"] <= 1.0
    assert 0.0 <= eval_out["controlled_benchmark"]["recall"] <= 1.0
    assert "per_scenario" in eval_out
