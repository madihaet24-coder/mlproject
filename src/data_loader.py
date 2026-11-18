"""Dataset acquisition, parsing, and hourly aggregation for UCI household power data."""

import io
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles downloading, parsing, and hourly aggregation of UCI electricity data."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.raw_zip_path = RAW_DATA_DIR / self.config.raw_zip_filename
        self.raw_txt_path = RAW_DATA_DIR / self.config.raw_txt_filename
        self._default_processed_path = PROCESSED_DATA_DIR / self.config.processed_filename
        self._root_processed_path = BASE_DIR / self.config.processed_filename

    @property
    def processed_csv_path(self) -> Path:
        """Return the active path to the processed hourly CSV file."""
        if self._default_processed_path.exists():
            return self._default_processed_path
        if self._root_processed_path.exists():
            return self._root_processed_path
        return self._default_processed_path

    def download_raw_data(self, force: bool = False) -> Path:
        """Download and extract the UCI Individual Household Electric Power Consumption dataset."""
        if self.raw_txt_path.exists() and not force:
            logger.info("Raw dataset already exists at %s", self.raw_txt_path)
            return self.raw_txt_path

        logger.info("Downloading UCI dataset from %s ...", self.config.uci_url)
        response = requests.get(self.config.uci_url, timeout=120)
        response.raise_for_status()

        with open(self.raw_zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(RAW_DATA_DIR)

        if not self.raw_txt_path.exists():
            txt_files = list(RAW_DATA_DIR.glob("*.txt"))
            if txt_files:
                txt_files[0].rename(self.raw_txt_path)
            else:
                raise FileNotFoundError("Extracted archive did not contain a .txt data file.")

        logger.info("Extracted raw dataset to %s", self.raw_txt_path)
        return self.raw_txt_path

    def load_raw_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """Parse minute-level semicolon-delimited UCI data with '?' missing-value handling."""
        path = Path(filepath) if filepath is not None else self.raw_txt_path
        if not path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at {path}")

        df = pd.read_csv(
            path,
            sep=";",
            na_values=["?", "nan", "NaN", ""],
            low_memory=False,
        )

        df[self.config.datetime_col] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        df = df.dropna(subset=[self.config.datetime_col])
        df = df.drop(columns=["Date", "Time"], errors="ignore")

        numeric_cols = [c for c in df.columns if c != self.config.datetime_col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(self.config.datetime_col).reset_index(drop=True)
        return df

    def aggregate_to_hourly(self, df_minute: pd.DataFrame) -> pd.DataFrame:
        """Aggregate minute-level kW readings into hourly energy consumption (kWh)."""
        dt_col = self.config.datetime_col
        target_col = self.config.target_col

        df_work = df_minute.copy()
        df_work[dt_col] = pd.to_datetime(df_work[dt_col])
        df_work["hour_bin"] = df_work[dt_col].dt.floor("h")

        numeric_cols = [c for c in df_work.columns if c not in (dt_col, "hour_bin")]
        agg_dict = {col: "mean" for col in numeric_cols}

        hourly = df_work.groupby("hour_bin", as_index=False).agg(agg_dict)
        valid_counts = (
            df_work.groupby("hour_bin")[target_col]
            .count()
            .rename("valid_minute_count")
            .reset_index()
        )
        hourly = hourly.merge(valid_counts, on="hour_bin", how="left")
        hourly = hourly.rename(columns={"hour_bin": dt_col})
        hourly = hourly.sort_values(dt_col).reset_index(drop=True)
        return hourly

    def process_and_save(self, force: bool = False) -> pd.DataFrame:
        """Load or compute the hourly dataset and ensure it is persisted in data/processed/."""
        if not force:
            if self._default_processed_path.exists():
                df = pd.read_csv(self._default_processed_path)
                df[self.config.datetime_col] = pd.to_datetime(df[self.config.datetime_col])
                return df
            if self._root_processed_path.exists():
                shutil.copy2(self._root_processed_path, self._default_processed_path)
                df = pd.read_csv(self._default_processed_path)
                df[self.config.datetime_col] = pd.to_datetime(df[self.config.datetime_col])
                return df

        if not self.raw_txt_path.exists() or force:
            self.download_raw_data(force=force)

        df_minute = self.load_raw_data(self.raw_txt_path)
        df_hourly = self.aggregate_to_hourly(df_minute)
        df_hourly.to_csv(self._default_processed_path, index=False)
        logger.info("Saved processed hourly dataset (%d rows) to %s", len(df_hourly), self._default_processed_path)
        return df_hourly
