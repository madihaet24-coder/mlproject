"""XGBoost regression model training, evaluation, and artifact persistence."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from src.config import MODELS_DIR, ModelConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, evaluates, saves, and loads a single reproducible XGBRegressor."""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config or ModelConfig()
        self.model: Optional[xgb.XGBRegressor] = None
        self.metadata: Dict[str, Any] = {}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Fit XGBRegressor with optional validation early stopping."""
        cfg = self.model_config
        use_early_stopping = X_val is not None and y_val is not None and len(X_val) > 0

        self.model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            early_stopping_rounds=cfg.early_stopping_rounds if use_early_stopping else None,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            objective="reg:squarederror",
            eval_metric="rmse",
        )

        if use_early_stopping:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_iter = getattr(self.model, "best_iteration", cfg.n_estimators)
        else:
            self.model.fit(X_train, y_train, verbose=False)
            best_iter = cfg.n_estimators

        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val) if use_early_stopping else {}

        self.metadata = {
            "model_type": "XGBRegressor",
            "best_iteration": int(best_iter) if best_iter is not None else cfg.n_estimators,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)) if X_val is not None else 0,
            "feature_names": list(X_train.columns),
            "hyperparameters": {
                "n_estimators": cfg.n_estimators,
                "learning_rate": cfg.learning_rate,
                "max_depth": cfg.max_depth,
                "subsample": cfg.subsample,
                "colsample_bytree": cfg.colsample_bytree,
                "early_stopping_rounds": cfg.early_stopping_rounds,
                "random_state": cfg.random_state,
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        return self.metadata

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate non-negative consumption predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute MAE, RMSE, and R^2 on a given partition."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=float)
        return {
            "mae": float(mean_absolute_error(y_arr, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_arr, preds))),
            "r2": float(r2_score(y_arr, preds)),
        }

    def save_model(self, name: str = "xgboost_forecaster") -> Path:
        """Persist trained model (.joblib) and metadata (.json) to models/."""
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized model.")
        model_path = MODELS_DIR / f"{name}.joblib"
        meta_path = MODELS_DIR / f"{name}_metadata.json"

        joblib.dump(self.model, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Saved model to %s and metadata to %s", model_path, meta_path)
        return model_path

    def load_model(self, name: str = "xgboost_forecaster") -> None:
        """Load persisted model and metadata from models/."""
        model_path = MODELS_DIR / f"{name}.joblib"
        meta_path = MODELS_DIR / f"{name}_metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        self.model = joblib.load(model_path)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
