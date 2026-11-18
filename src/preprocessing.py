"""Data cleaning, missing-value imputation, and strict chronological splitting."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DataConfig

logger = logging.getLogger(__name__)


class Preprocessor:
    """Cleans hourly electricity data and enforces leak-free chronological splits."""

    def __init__(self, data_config: Optional[DataConfig] = None):
        self.data_config = data_config or DataConfig()

    def clean_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort chronologically, deduplicate timestamps, and handle missing values."""
        dt_col = self.data_config.datetime_col
        target_col = self.data_config.target_col

        cleaned = df.copy()
        cleaned[dt_col] = pd.to_datetime(cleaned[dt_col])
        cleaned = cleaned.sort_values(dt_col).drop_duplicates(subset=[dt_col]).reset_index(drop=True)

        # Invalidate hours with insufficient minute observations if metadata is present
        if "valid_minute_count" in cleaned.columns:
            insufficient_mask = cleaned["valid_minute_count"] < self.data_config.min_hourly_observations
            if insufficient_mask.any():
                cleaned.loc[insufficient_mask, target_col] = np.nan

        # Ensure physical non-negativity before interpolation
        cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")
        cleaned.loc[cleaned[target_col] < 0, target_col] = np.nan

        # Interpolate internal gaps and backfill/forward-fill boundary nulls
        if cleaned[target_col].isna().any():
            cleaned[target_col] = (
                cleaned[target_col]
                .interpolate(method="linear", limit_direction="both")
                .bfill()
                .ffill()
            )

        return cleaned

    def split_chronological(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train, validation, and test partitions."""
        dt_col = self.data_config.datetime_col
        ordered = df.sort_values(dt_col).reset_index(drop=True)
        n = len(ordered)
        if n < 10:
            raise ValueError(f"Dataset too small to split chronologically ({n} rows).")

        train_end = int(n * self.data_config.train_ratio)
        val_end = train_end + int(n * self.data_config.val_ratio)

        df_train = ordered.iloc[:train_end].copy().reset_index(drop=True)
        df_val = ordered.iloc[train_end:val_end].copy().reset_index(drop=True)
        df_test = ordered.iloc[val_end:].copy().reset_index(drop=True)

        return df_train, df_val, df_test
