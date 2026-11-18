"""Unit tests for src.model_training."""

import numpy as np
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer


def test_model_training_and_metadata(synthetic_hourly_df):
    engineer = FeatureEngineer()
    df_feats = engineer.create_features(synthetic_hourly_df)
    X, y, _ = engineer.get_feature_matrix(df_feats)

    trainer = ModelTrainer()
    meta = trainer.train(X.iloc[:350], y.iloc[:350], X.iloc[350:450], y.iloc[350:450])
    assert meta["train_samples"] == 350
    assert meta["val_samples"] == 100
    assert len(meta["feature_names"]) == 22


def test_model_predictions_non_negative_and_metrics(synthetic_hourly_df):
    engineer = FeatureEngineer()
    df_feats = engineer.create_features(synthetic_hourly_df)
    X, y, _ = engineer.get_feature_matrix(df_feats)

    trainer = ModelTrainer()
    trainer.train(X.iloc[:350], y.iloc[:350], X.iloc[350:450], y.iloc[350:450])
    preds = trainer.predict(X.iloc[450:])
    assert (preds >= 0).all()

    metrics = trainer.evaluate(X.iloc[450:], y.iloc[450:])
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]


def test_model_persistence_roundtrip(synthetic_hourly_df, tmp_path, monkeypatch):
    import src.model_training as mt

    monkeypatch.setattr(mt, "MODELS_DIR", tmp_path)
    engineer = FeatureEngineer()
    df_feats = engineer.create_features(synthetic_hourly_df)
    X, y, _ = engineer.get_feature_matrix(df_feats)

    trainer1 = ModelTrainer()
    trainer1.train(X.iloc[:350], y.iloc[:350], X.iloc[350:450], y.iloc[350:450])
    preds1 = trainer1.predict(X.iloc[450:])
    trainer1.save_model("test_model")

    trainer2 = ModelTrainer()
    trainer2.load_model("test_model")
    preds2 = trainer2.predict(X.iloc[450:])
    np.testing.assert_allclose(preds1, preds2, atol=1e-6)
