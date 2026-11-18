"""Unit tests for src.preprocessing."""

import numpy as np
import pandas as pd
from src.preprocessing import Preprocessor


def test_clean_hourly_data_sorts_and_deduplicates(synthetic_hourly_df):
    prep = Preprocessor()
    shuffled = pd.concat([synthetic_hourly_df.iloc[::-1], synthetic_hourly_df.iloc[:3]], ignore_index=True)
    cleaned = prep.clean_hourly_data(shuffled)
    assert len(cleaned) == len(synthetic_hourly_df)
    assert cleaned["Datetime"].is_monotonic_increasing
    assert not cleaned["Datetime"].duplicated().any()


def test_clean_hourly_data_handles_low_valid_minutes_and_nulls(synthetic_hourly_df):
    prep = Preprocessor()
    df = synthetic_hourly_df.copy()
    df.loc[10, "valid_minute_count"] = 15
    df.loc[20, "Global_active_power"] = np.nan
    cleaned = prep.clean_hourly_data(df)
    assert not cleaned["Global_active_power"].isna().any()
    assert (cleaned["Global_active_power"] >= 0).all()


def test_split_chronological_zero_overlap(synthetic_hourly_df):
    prep = Preprocessor()
    df_train, df_val, df_test = prep.split_chronological(synthetic_hourly_df)
    assert len(df_train) + len(df_val) + len(df_test) == len(synthetic_hourly_df)
    assert df_train["Datetime"].max() < df_val["Datetime"].min()
    assert df_val["Datetime"].max() < df_test["Datetime"].min()
