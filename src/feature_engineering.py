"""Leak-free lag, rolling, and cyclical temporal feature engineering."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DataConfig, FeatureConfig


class FeatureEngineer:
    """Constructs 22 strictly leak-free temporal, lag, and rolling features."""

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        data_config: Optional[DataConfig] = None,
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.data_config = data_config or DataConfig()
        self.feature_names: List[str] = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        for lag in self.feature_config.lags:
            names.append(f"lag_{lag}h")
        for win in self.feature_config.rolling_windows:
            names.append(f"rolling_mean_{win}h")
            names.append(f"rolling_std_{win}h")
        names.extend(
            [
                "hour",
                "day_of_week",
                "month",
                "is_weekend",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
            ]
        )
        return names

    def create_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Create lag, rolling (past-only), and cyclical calendar features."""
        dt_col = self.data_config.datetime_col
        target_col = self.data_config.target_col

        feat_df = df.copy()
        feat_df[dt_col] = pd.to_datetime(feat_df[dt_col])
        feat_df = feat_df.sort_values(dt_col).reset_index(drop=True)

        # 1. Autoregressive lags (strictly >= 1 hour in the past)
        for lag in self.feature_config.lags:
            feat_df[f"lag_{lag}h"] = feat_df[target_col].shift(lag)

        # 2. Lagged rolling statistics (shifted by 1 so target_t is never included)
        shifted_target = feat_df[target_col].shift(1)
        for win in self.feature_config.rolling_windows:
            feat_df[f"rolling_mean_{win}h"] = shifted_target.rolling(window=win, min_periods=win).mean()
            feat_df[f"rolling_std_{win}h"] = shifted_target.rolling(window=win, min_periods=win).std().fillna(0.0)

        # 3. Calendar and cyclical encodings (known at prediction time t)
        dt_series = feat_df[dt_col]
        feat_df["hour"] = dt_series.dt.hour
        feat_df["day_of_week"] = dt_series.dt.dayofweek
        feat_df["month"] = dt_series.dt.month
        feat_df["is_weekend"] = (feat_df["day_of_week"] >= 5).astype(int)

        feat_df["hour_sin"] = np.sin(2.0 * np.pi * feat_df["hour"] / 24.0)
        feat_df["hour_cos"] = np.cos(2.0 * np.pi * feat_df["hour"] / 24.0)
        feat_df["dow_sin"] = np.sin(2.0 * np.pi * feat_df["day_of_week"] / 7.0)
        feat_df["dow_cos"] = np.cos(2.0 * np.pi * feat_df["day_of_week"] / 7.0)

        if drop_na:
            feat_df = feat_df.dropna(subset=self.feature_names + [target_col]).reset_index(drop=True)

        return feat_df

    def get_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract (X, y, timestamps) from a feature-engineered DataFrame."""
        X = df[self.feature_names].copy()
        y = df[self.data_config.target_col].copy()
        timestamps = pd.to_datetime(df[self.data_config.datetime_col]).copy()
        return X, y, timestamps

    @staticmethod
    def forecast_recursive_24h(
        df_history: pd.DataFrame,
        model_trainer,
        start_idx: int,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """Perform autoregressive recursive multi-step forecasting from start_idx."""
        engineer = FeatureEngineer()
        dt_col = engineer.data_config.datetime_col
        target_col = engineer.data_config.target_col

        max_lag = max(
            max(engineer.feature_config.lags),
            max(engineer.feature_config.rolling_windows) + 1,
        )
        if start_idx < max_lag:
            raise ValueError(f"start_idx ({start_idx}) must be >= {max_lag} to build lag features.")

        end_idx = min(start_idx + horizon_hours, len(df_history))
        history_buf = df_history.iloc[start_idx - max_lag : start_idx][[dt_col, target_col]].copy()
        history_buf[dt_col] = pd.to_datetime(history_buf[dt_col])
        history_buf[target_col] = history_buf[target_col].ffill().bfill()

        rows = []
        for idx in range(start_idx, end_idx):
            ts = pd.to_datetime(df_history.iloc[idx][dt_col])
            actual_val = df_history.iloc[idx][target_col]

            # Append placeholder row for step t and compute features using past buffer
            step_row = pd.DataFrame({dt_col: [ts], target_col: [np.nan]})
            work_df = pd.concat([history_buf, step_row], ignore_index=True)
            feats_df = engineer.create_features(work_df, drop_na=False)
            X_step = feats_df.iloc[[-1]][engineer.feature_names]

            pred_val = float(max(0.0, model_trainer.predict(X_step)[0]))
            rows.append(
                {
                    "timestamp": ts,
                    "actual_kwh": float(actual_val) if pd.notna(actual_val) else np.nan,
                    "forecast_kwh": pred_val,
                }
            )

            # Feed prediction back into history buffer
            history_buf = pd.concat(
                [history_buf, pd.DataFrame({dt_col: [ts], target_col: [pred_val]})],
                ignore_index=True,
            ).iloc[-max_lag:]

        return pd.DataFrame(rows)
