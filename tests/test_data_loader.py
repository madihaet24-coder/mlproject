"""Unit tests for src.data_loader."""

import pandas as pd
from src.data_loader import DataLoader


def test_load_raw_data_handles_question_mark_nulls(synthetic_raw_txt):
    loader = DataLoader()
    df = loader.load_raw_data(synthetic_raw_txt)
    assert len(df) == 4
    assert pd.isna(df.loc[1, "Global_active_power"])
    assert pd.api.types.is_datetime64_any_dtype(df["Datetime"])


def test_aggregate_to_hourly_computes_mean_and_valid_count(synthetic_raw_txt):
    loader = DataLoader()
    df_min = loader.load_raw_data(synthetic_raw_txt)
    df_hr = loader.aggregate_to_hourly(df_min)
    assert len(df_hr) == 2
    assert df_hr.loc[0, "valid_minute_count"] == 2
    assert abs(df_hr.loc[0, "Global_active_power"] - 3.6) < 1e-6
    assert df_hr.loc[1, "valid_minute_count"] == 1


def test_processed_csv_path_exists():
    loader = DataLoader()
    assert loader.processed_csv_path.exists()
