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


def test_ingest_custom_dataset_hourly_and_alias_columns(tmp_path, monkeypatch, synthetic_hourly_df):
    import src.data_loader as dl

    custom_dir = tmp_path / "custom"
    active_file = tmp_path / "active_dataset.json"
    monkeypatch.setattr(dl, "CUSTOM_DATA_DIR", custom_dir)
    monkeypatch.setattr(dl, "ACTIVE_DATASET_FILE", active_file)

    # Create a CSV with non-standard column aliases ('timestamp' and 'power_kwh')
    csv_path = tmp_path / "my_home_meter.csv"
    df_custom = pd.DataFrame(
        {
            "timestamp": synthetic_hourly_df["Datetime"],
            "power_kwh": synthetic_hourly_df["Global_active_power"],
        }
    )
    df_custom.to_csv(csv_path, index=False)

    loader = DataLoader()
    entry = loader.ingest_custom_dataset(
        source_path=csv_path,
        dataset_name="Home Meter 2025",
        activate=True,
    )
    assert entry["name"] == "Home Meter 2025"
    assert entry["rows"] == len(synthetic_hourly_df)
    assert entry["source_target_col"] == "power_kwh"
    assert loader.processed_csv_path == custom_dir / f"{entry['id']}.csv"

    # Switching back to uci_default updates active dataset
    reg = loader.activate_dataset("uci_default")
    assert reg["active_id"] == "uci_default"


def test_ingest_custom_dataset_rejects_too_short_series(tmp_path, monkeypatch, synthetic_hourly_df):
    import pytest
    import src.data_loader as dl

    monkeypatch.setattr(dl, "CUSTOM_DATA_DIR", tmp_path / "custom")
    monkeypatch.setattr(dl, "ACTIVE_DATASET_FILE", tmp_path / "active_dataset.json")

    short_csv = tmp_path / "too_short.csv"
    synthetic_hourly_df.iloc[:100].to_csv(short_csv, index=False)

    loader = DataLoader()
    with pytest.raises(ValueError, match="at least 300 hourly records"):
        loader.ingest_custom_dataset(source_path=short_csv)
