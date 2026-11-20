"""Dataset acquisition, parsing, hourly aggregation, and custom dataset registry."""

import io
import json
import logging
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    ACTIVE_DATASET_FILE,
    BASE_DIR,
    CUSTOM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    DataConfig,
)

logger = logging.getLogger(__name__)

_DATETIME_ALIASES = ("datetime", "timestamp", "date_time", "date", "time", "dt", "hour")
_TARGET_ALIASES = (
    "global_active_power",
    "active_power",
    "consumption",
    "power",
    "kwh",
    "energy",
    "load",
    "usage",
    "demand",
    "target",
    "value",
)


class DataLoader:
    """Handles downloading, parsing, custom dataset ingestion, and dataset switching."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.raw_zip_path = RAW_DATA_DIR / self.config.raw_zip_filename
        self.raw_txt_path = RAW_DATA_DIR / self.config.raw_txt_filename
        self._default_processed_path = PROCESSED_DATA_DIR / self.config.processed_filename
        self._root_processed_path = BASE_DIR / self.config.processed_filename

    @property
    def processed_csv_path(self) -> Path:
        """Return the active path to the processed hourly CSV file."""
        if self.config.custom_csv_path:
            return Path(self.config.custom_csv_path)
        if ACTIVE_DATASET_FILE.exists():
            try:
                registry = json.loads(ACTIVE_DATASET_FILE.read_text(encoding="utf-8"))
                active_path = registry.get("active_path")
                if active_path and Path(active_path).exists():
                    return Path(active_path)
            except Exception:
                pass
        if self._default_processed_path.exists():
            return self._default_processed_path
        if self._root_processed_path.exists():
            return self._root_processed_path
        return self._default_processed_path

    def get_default_uci_path(self) -> Path:
        """Ensure the default UCI hourly CSV exists in data/processed/ and return its path."""
        if not self._default_processed_path.exists() and self._root_processed_path.exists():
            shutil.copy2(self._root_processed_path, self._default_processed_path)
        return self._default_processed_path

    def _summarize_csv(self, csv_path: Path) -> Dict[str, Any]:
        df_meta = pd.read_csv(csv_path, usecols=[self.config.datetime_col])
        dt_series = pd.to_datetime(df_meta[self.config.datetime_col], errors="coerce").dropna()
        return {
            "rows": int(len(df_meta)),
            "start_date": str(dt_series.min()) if not dt_series.empty else "",
            "end_date": str(dt_series.max()) if not dt_series.empty else "",
        }

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        ACTIVE_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_DATASET_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def list_datasets(self) -> Dict[str, Any]:
        """Return the dataset registry including the default UCI and any custom datasets."""
        default_path = self.get_default_uci_path()
        registry: Dict[str, Any] = {}
        if ACTIVE_DATASET_FILE.exists():
            try:
                registry = json.loads(ACTIVE_DATASET_FILE.read_text(encoding="utf-8"))
            except Exception:
                registry = {}

        datasets: List[Dict[str, Any]] = [
            d for d in registry.get("datasets", []) if Path(d.get("path", "")).exists()
        ]

        if not any(d.get("id") == "uci_default" for d in datasets) and default_path.exists():
            meta = self._summarize_csv(default_path)
            datasets.insert(
                0,
                {
                    "id": "uci_default",
                    "name": "UCI Household Electric Power (Default)",
                    "path": str(default_path.resolve()),
                    "rows": meta["rows"],
                    "start_date": meta["start_date"],
                    "end_date": meta["end_date"],
                    "is_default": True,
                    "added_at": "2006-12-16T17:00:00",
                },
            )

        active_id = registry.get("active_id", "uci_default")
        active_entry = next((d for d in datasets if d["id"] == active_id), None)
        if active_entry is None and datasets:
            active_entry = datasets[0]

        out = {
            "active_id": active_entry["id"] if active_entry else "uci_default",
            "active_name": active_entry["name"] if active_entry else "UCI Household Electric Power (Default)",
            "active_path": active_entry["path"] if active_entry else str(default_path.resolve()),
            "datasets": datasets,
        }
        self._save_registry(out)
        return out

    def activate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Switch the active dataset used by validation, training, and evaluation."""
        registry = self.list_datasets()
        match = next((d for d in registry["datasets"] if d["id"] == dataset_id), None)
        if match is None:
            raise KeyError(f"Dataset ID '{dataset_id}' not found in registry.")

        registry["active_id"] = match["id"]
        registry["active_name"] = match["name"]
        registry["active_path"] = match["path"]
        self._save_registry(registry)
        logger.info("Activated dataset '%s' (%s)", match["name"], match["path"])
        return registry

    @staticmethod
    def _detect_separator(source_path: Path) -> str:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
        return ";" if header.count(";") > header.count(",") else ","

    @staticmethod
    def _resolve_datetime_series(
        df: pd.DataFrame, datetime_col: Optional[str] = None
    ) -> pd.Series:
        if datetime_col:
            if datetime_col not in df.columns:
                raise ValueError(
                    f"Specified datetime column '{datetime_col}' not found in {list(df.columns)}."
                )
            parsed = pd.to_datetime(df[datetime_col], errors="coerce")
            if parsed.notna().mean() < 0.8:
                raise ValueError(f"Column '{datetime_col}' could not be parsed as timestamps.")
            return parsed

        lower_map = {str(c).strip().lower(): c for c in df.columns}
        if "date" in lower_map and "time" in lower_map:
            combined = (
                df[lower_map["date"]].astype(str) + " " + df[lower_map["time"]].astype(str)
            )
            parsed = pd.to_datetime(combined, dayfirst=True, format="mixed", errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return parsed

        for alias in _DATETIME_ALIASES:
            if alias in lower_map:
                col_name = lower_map[alias]
                parsed = pd.to_datetime(df[col_name], format="mixed", errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    return parsed

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return parsed

        raise ValueError(
            f"Could not detect a valid timestamp column in {list(df.columns)}. "
            "Provide a column such as 'Datetime' or 'timestamp'."
        )

    @staticmethod
    def _resolve_target_series(
        df: pd.DataFrame, target_col: Optional[str] = None, exclude_cols: tuple[str, ...] = ()
    ) -> tuple[str, pd.Series]:
        if target_col:
            if target_col not in df.columns:
                raise ValueError(
                    f"Specified target column '{target_col}' not found in {list(df.columns)}."
                )
            series = pd.to_numeric(df[target_col], errors="coerce")
            if series.notna().sum() == 0:
                raise ValueError(f"Target column '{target_col}' contains no numeric values.")
            return target_col, series

        lower_map = {
            str(c).strip().lower(): c for c in df.columns if c not in exclude_cols
        }
        for alias in _TARGET_ALIASES:
            if alias in lower_map:
                col_name = lower_map[alias]
                series = pd.to_numeric(df[col_name], errors="coerce")
                if series.notna().mean() >= 0.5:
                    return col_name, series

        for col in df.columns:
            if col in exclude_cols or str(col).strip().lower() == "valid_minute_count":
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().mean() >= 0.5:
                return col, series

        raise ValueError(
            f"Could not detect a numeric consumption column in {list(df.columns)}."
        )

    def ingest_custom_dataset(
        self,
        source_path: Path,
        dataset_name: Optional[str] = None,
        datetime_col: Optional[str] = None,
        target_col: Optional[str] = None,
        activate: bool = True,
    ) -> Dict[str, Any]:
        """Validate, standardize, aggregate if sub-hourly, and register a custom dataset."""
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Uploaded dataset file not found at {src}")

        sep = self._detect_separator(src)
        df_raw = pd.read_csv(
            src,
            sep=sep,
            na_values=["?", "nan", "NaN", "", "null", "NULL"],
            low_memory=False,
        )
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        dt_series = self._resolve_datetime_series(df_raw, datetime_col=datetime_col)
        used_dt_col = datetime_col if datetime_col else None
        exclude = (used_dt_col,) if used_dt_col else ("Date", "Time", "Datetime", "timestamp")
        resolved_target_name, target_series = self._resolve_target_series(
            df_raw, target_col=target_col, exclude_cols=exclude
        )

        standardized = pd.DataFrame(
            {
                self.config.datetime_col: dt_series,
                self.config.target_col: target_series,
            }
        )
        if "valid_minute_count" in df_raw.columns:
            standardized["valid_minute_count"] = pd.to_numeric(
                df_raw["valid_minute_count"], errors="coerce"
            ).fillna(60).astype(int)

        standardized = (
            standardized.dropna(subset=[self.config.datetime_col])
            .sort_values(self.config.datetime_col)
            .reset_index(drop=True)
        )

        if standardized.empty:
            raise ValueError("Uploaded file has zero rows with valid timestamps.")

        # Aggregate sub-hourly records to hourly if multiple observations fall in the same hour
        hour_bins = standardized[self.config.datetime_col].dt.floor("h")
        if hour_bins.duplicated().any():
            df_hourly = self.aggregate_to_hourly(
                standardized.drop(columns=["valid_minute_count"], errors="ignore")
            )
            # Scale valid_minute_count if sub-hourly intervals are < 60 per hour (e.g. 15-min or 30-min)
            obs_per_hour = int(hour_bins.value_counts().median())
            if 1 < obs_per_hour < 60:
                df_hourly["valid_minute_count"] = np.where(
                    df_hourly[self.config.target_col].notna(), 60, 0
                )
        else:
            standardized[self.config.datetime_col] = hour_bins
            standardized = standardized.drop_duplicates(subset=[self.config.datetime_col])
            full_grid = pd.DataFrame(
                {
                    self.config.datetime_col: pd.date_range(
                        standardized[self.config.datetime_col].min(),
                        standardized[self.config.datetime_col].max(),
                        freq="1h",
                    )
                }
            )
            df_hourly = full_grid.merge(
                standardized, on=self.config.datetime_col, how="left"
            )
            if "valid_minute_count" not in df_hourly.columns:
                df_hourly["valid_minute_count"] = np.where(
                    df_hourly[self.config.target_col].notna(), 60, 0
                )
            else:
                df_hourly["valid_minute_count"] = (
                    df_hourly["valid_minute_count"].fillna(0).astype(int)
                )

        if len(df_hourly) < 300:
            raise ValueError(
                f"Custom dataset contains {len(df_hourly)} hourly records; "
                "at least 300 hourly records are required for 168h lag features and train/val/test splitting."
            )

        clean_name = (dataset_name or src.stem).strip() or "Custom Electricity Dataset"
        slug = re.sub(r"[^a-z0-9]+", "_", clean_name.lower()).strip("_")[:32] or "dataset"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"custom_{slug}_{stamp}"
        dest_path = CUSTOM_DATA_DIR / f"{dataset_id}.csv"
        df_hourly.to_csv(dest_path, index=False)

        registry = self.list_datasets()
        entry = {
            "id": dataset_id,
            "name": clean_name,
            "path": str(dest_path.resolve()),
            "rows": int(len(df_hourly)),
            "start_date": str(df_hourly[self.config.datetime_col].min()),
            "end_date": str(df_hourly[self.config.datetime_col].max()),
            "source_target_col": resolved_target_name,
            "is_default": False,
            "added_at": datetime.now().isoformat(),
        }
        registry["datasets"].append(entry)
        if activate:
            registry["active_id"] = entry["id"]
            registry["active_name"] = entry["name"]
            registry["active_path"] = entry["path"]
        self._save_registry(registry)

        logger.info(
            "Ingested custom dataset '%s' (%d hourly rows) -> %s",
            clean_name,
            len(df_hourly),
            dest_path,
        )
        return entry

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
        """Load the active processed hourly dataset or build UCI dataset from raw files."""
        if not force:
            active_path = self.processed_csv_path
            if active_path.exists():
                if (
                    active_path == self._root_processed_path
                    and not self._default_processed_path.exists()
                ):
                    shutil.copy2(self._root_processed_path, self._default_processed_path)
                    active_path = self._default_processed_path
                df = pd.read_csv(active_path)
                df[self.config.datetime_col] = pd.to_datetime(df[self.config.datetime_col])
                return df

        if not self.raw_txt_path.exists() or force:
            self.download_raw_data(force=force)

        df_minute = self.load_raw_data(self.raw_txt_path)
        df_hourly = self.aggregate_to_hourly(df_minute)
        df_hourly.to_csv(self._default_processed_path, index=False)
        logger.info(
            "Saved processed hourly dataset (%d rows) to %s",
            len(df_hourly),
            self._default_processed_path,
        )
        return df_hourly
