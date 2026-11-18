"""Shared pytest fixtures for unit and integration tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_hourly_df() -> pd.DataFrame:
    """Create a 720-hour (30-day) synthetic hourly electricity consumption DataFrame."""
    rng = np.random.default_rng(42)
    n_hours = 720
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=n_hours, freq="1h")
    hours = timestamps.hour.to_numpy()
    diurnal = 1.2 + 0.8 * np.sin(2 * np.pi * (hours - 7) / 24.0)
    noise = rng.normal(0.0, 0.12, size=n_hours)
    power = np.clip(diurnal + noise, 0.2, None)

    return pd.DataFrame(
        {
            "Datetime": timestamps,
            "Global_active_power": power,
            "Global_reactive_power": rng.uniform(0.05, 0.25, size=n_hours),
            "Voltage": rng.normal(235.0, 1.5, size=n_hours),
            "Global_intensity": power * 4.2,
            "Sub_metering_1": rng.uniform(0.0, 1.0, size=n_hours),
            "Sub_metering_2": rng.uniform(0.0, 1.0, size=n_hours),
            "Sub_metering_3": rng.uniform(0.0, 15.0, size=n_hours),
            "valid_minute_count": np.full(n_hours, 60, dtype=int),
        }
    )


@pytest.fixture
def synthetic_raw_txt(tmp_path) -> str:
    """Create a small UCI-format semicolon-delimited raw text file with '?' missing values."""
    lines = [
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3",
        "16/12/2006;17:00:00;3.500;0.200;234.50;15.0;0.0;0.0;16.0",
        "16/12/2006;17:01:00;?;0.210;234.60;?;0.0;0.0;16.0",
        "16/12/2006;17:02:00;3.700;0.220;234.40;15.8;0.0;0.0;16.0",
        "16/12/2006;18:00:00;2.400;0.100;235.00;10.2;0.0;0.0;12.0",
    ]
    raw_file = tmp_path / "sample_power.txt"
    raw_file.write_text("\n".join(lines), encoding="utf-8")
    return str(raw_file)
