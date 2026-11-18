"""Centralized configuration dataclasses and directory paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass
class DataConfig:
    """Configuration for data ingestion, cleaning, and chronological splitting."""

    datetime_col: str = "Datetime"
    target_col: str = "Global_active_power"
    min_hourly_observations: int = 40
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    uci_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
        "household_power_consumption.zip"
    )
    raw_zip_filename: str = "household_power_consumption.zip"
    raw_txt_filename: str = "household_power_consumption.txt"
    processed_filename: str = "hourly_electricity_consumption.csv"


@dataclass
class FeatureConfig:
    """Configuration for leak-free lag, rolling, and cyclical temporal features."""

    lags: Tuple[int, ...] = (1, 2, 3, 24, 48, 168)
    rolling_windows: Tuple[int, ...] = (3, 6, 24, 168)
    target_col: str = "Global_active_power"
    datetime_col: str = "Datetime"


@dataclass
class ModelConfig:
    """Hyperparameters for the single XGBoost forecasting model."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 25
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class AnomalyDetectorConfig:
    """Configuration for the Adaptive Rolling Median + MAD anomaly detector."""

    sensitivity: float = 3.0
    window_size: int = 168
    min_history: int = 24
    eps: float = 1e-4
    persistence: int = 1
    exclude_anomalies: bool = True


@dataclass
class SyntheticScenarioConfig:
    """Configuration for controlled synthetic anomaly injection benchmarks."""

    random_seed: int = 42
    n_temp_spikes: int = 4
    n_repeated_spikes: int = 3
    sustained_shift_hours: int = 36
    sudden_drop_hours: int = 12
    routine_change_hours: int = 240
