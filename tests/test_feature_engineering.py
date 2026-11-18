"""Unit tests for src.feature_engineering."""

import numpy as np
import pandas as pd
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer


def test_feature_count_is_22(synthetic_hourly_df):
    engineer = FeatureEngineer()
    assert len(engineer.feature_names) == 22
    df_feats = engineer.create_features(synthetic_hourly_df)
    X, y, ts = engineer.get_feature_matrix(df_feats)
    assert X.shape[1] == 22
    assert len(X) == len(y) == len(ts)
    assert not X.isna().any().any()


def test_zero_future_target_leakage(synthetic_hourly_df):
    engineer = FeatureEngineer()
    cutoff = 400
    df_orig = engineer.create_features(synthetic_hourly_df)

    df_perturbed = synthetic_hourly_df.copy()
    df_perturbed.loc[cutoff:, "Global_active_power"] = 99.0
    df_pert_feats = engineer.create_features(df_perturbed)

    orig_before = df_orig[df_orig["Datetime"] < synthetic_hourly_df.loc[cutoff, "Datetime"]]
    pert_before = df_pert_feats[df_pert_feats["Datetime"] < synthetic_hourly_df.loc[cutoff, "Datetime"]]

    pd.testing.assert_frame_equal(
        orig_before[engineer.feature_names].reset_index(drop=True),
        pert_before[engineer.feature_names].reset_index(drop=True),
    )


def test_rolling_features_exclude_current_target(synthetic_hourly_df):
    engineer = FeatureEngineer()
    df_a = synthetic_hourly_df.copy()
    df_b = synthetic_hourly_df.copy()
    target_idx = 300
    df_b.loc[target_idx, "Global_active_power"] = 50.0

    feats_a = engineer.create_features(df_a)
    feats_b = engineer.create_features(df_b)

    ts_target = synthetic_hourly_df.loc[target_idx, "Datetime"]
    row_a = feats_a[feats_a["Datetime"] == ts_target][engineer.feature_names].iloc[0]
    row_b = feats_b[feats_b["Datetime"] == ts_target][engineer.feature_names].iloc[0]
    pd.testing.assert_series_equal(row_a, row_b)


def test_forecast_recursive_24h_shape(synthetic_hourly_df):
    engineer = FeatureEngineer()
    df_feats = engineer.create_features(synthetic_hourly_df)
    X, y, _ = engineer.get_feature_matrix(df_feats)

    trainer = ModelTrainer()
    trainer.train(X.iloc[:300], y.iloc[:300], X.iloc[300:400], y.iloc[300:400])

    sim_df = FeatureEngineer.forecast_recursive_24h(
        df_history=synthetic_hourly_df,
        model_trainer=trainer,
        start_idx=500,
        horizon_hours=24,
    )
    assert len(sim_df) == 24
    assert list(sim_df.columns) == ["timestamp", "actual_kwh", "forecast_kwh"]
    assert (sim_df["forecast_kwh"] >= 0).all()
