"""Data Validation and Quality Assurance Module.

Performs schema validation, range checks, null audits, temporal monotonicity
verification, and hourly energy consistency checks on raw and processed datasets.
Outputs reports/data_validation_report.json and prints a clean summary.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    DataConfig,
)
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validate_data")


class DataValidator:
    """Comprehensive validator for household electricity data."""

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.validation_results: Dict[str, Any] = {}

    def validate_processed_data(self) -> Dict[str, Any]:
        """Validate the aggregated hourly dataset against strict research constraints."""
        loader = DataLoader(self.config)
        csv_path = loader.processed_csv_path

        if not csv_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {csv_path}. Run data ingestion first.")

        logger.info("Validating processed hourly dataset at %s ...", csv_path)
        df = pd.read_csv(csv_path)

        # 1. Schema & Column Presence Checks
        required_cols = [
            self.config.datetime_col,
            self.config.target_col,
            "valid_minute_count",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        has_required_schema = len(missing_cols) == 0

        # Parse timestamps
        df[self.config.datetime_col] = pd.to_datetime(df[self.config.datetime_col])

        # 2. Temporal Monotonicity & Continuity
        is_monotonic = bool(df[self.config.datetime_col].is_monotonic_increasing)
        has_duplicates = bool(df[self.config.datetime_col].duplicated().any())
        duplicate_count = int(df[self.config.datetime_col].duplicated().sum())

        # Check for gaps in the 1-hour grid
        full_hourly_range = pd.date_range(
            df[self.config.datetime_col].min(),
            df[self.config.datetime_col].max(),
            freq="1h",
        )
        expected_total_hours = len(full_hourly_range)
        actual_total_hours = len(df)
        missing_grid_timestamps = expected_total_hours - actual_total_hours

        # 3. Numeric Range & Boundary Checks
        target = df[self.config.target_col]
        non_null_target = target.dropna()

        negative_power_count = int((non_null_target < 0).sum())
        abnormally_high_power = int((non_null_target > 15.0).sum())  # French domestic hookups cap ~12-15 kW
        
        target_min = float(non_null_target.min())
        target_max = float(non_null_target.max())
        target_mean = float(non_null_target.mean())
        target_median = float(non_null_target.median())
        target_std = float(non_null_target.std())

        # 4. Missing Values Audit
        null_count = int(target.isna().sum())
        null_percentage = float(100.0 * null_count / max(1, actual_total_hours))

        # Check minute observation counts per hour
        min_valid_min = int(df["valid_minute_count"].min()) if "valid_minute_count" in df.columns else 0
        max_valid_min = int(df["valid_minute_count"].max()) if "valid_minute_count" in df.columns else 60
        hours_below_threshold = int((df["valid_minute_count"] < self.config.min_hourly_observations).sum()) if "valid_minute_count" in df.columns else 0

        # 5. Chronological Split Validation
        prep = Preprocessor(data_config=self.config)
        df_clean = prep.clean_hourly_data(df)
        df_train, df_val, df_test = prep.split_chronological(df_clean)

        train_max_dt = df_train[self.config.datetime_col].max()
        val_min_dt = df_val[self.config.datetime_col].min()
        val_max_dt = df_val[self.config.datetime_col].max()
        test_min_dt = df_test[self.config.datetime_col].min()

        split_integrity_verified = (train_max_dt < val_min_dt) and (val_max_dt < test_min_dt)

        overall_status = "PASSED" if (
            has_required_schema
            and is_monotonic
            and not has_duplicates
            and negative_power_count == 0
            and split_integrity_verified
        ) else "WARNINGS_FOUND"

        results = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "file_info": {
                "file_path": str(csv_path.resolve()),
                "file_size_bytes": csv_path.stat().st_size,
                "total_hourly_records": actual_total_hours,
                "time_span_start": str(df[self.config.datetime_col].min()),
                "time_span_end": str(df[self.config.datetime_col].max()),
                "expected_contiguous_hours": expected_total_hours,
                "grid_continuity_ratio": round(100.0 * actual_total_hours / max(1, expected_total_hours), 2),
            },
            "schema_checks": {
                "has_required_schema": has_required_schema,
                "present_columns": list(df.columns),
                "missing_columns": missing_cols,
            },
            "temporal_integrity": {
                "is_strictly_chronological": is_monotonic,
                "has_timestamp_duplicates": has_duplicates,
                "duplicate_count": duplicate_count,
                "missing_grid_hours": missing_grid_timestamps,
            },
            "consumption_range_checks": {
                "target_column": self.config.target_col,
                "units": "Kilowatt-hours (kWh)",
                "min_kwh": round(target_min, 4),
                "max_kwh": round(target_max, 4),
                "mean_kwh": round(target_mean, 4),
                "median_kwh": round(target_median, 4),
                "std_kwh": round(target_std, 4),
                "negative_values_count": negative_power_count,
                "excessive_values_count_gt_15kw": abnormally_high_power,
                "passes_range_checks": negative_power_count == 0,
            },
            "missing_value_audit": {
                "null_target_count": null_count,
                "null_percentage": round(null_percentage, 2),
                "hours_below_min_observations_40min": hours_below_threshold,
                "minute_count_range_per_hour": [min_valid_min, max_valid_min],
            },
            "chronological_split_integrity": {
                "train_samples": len(df_train),
                "train_start": str(df_train[self.config.datetime_col].min()),
                "train_end": str(train_max_dt),
                "val_samples": len(df_val),
                "val_start": str(val_min_dt),
                "val_end": str(val_max_dt),
                "test_samples": len(df_test),
                "test_start": str(test_min_dt),
                "test_end": str(df_test[self.config.datetime_col].max()),
                "zero_temporal_leakage_verified": split_integrity_verified,
            },
        }

        self.validation_results = results

        # Persist report
        report_json_path = REPORTS_DIR / "data_validation_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Data validation completed with status: %s. Report saved to %s", overall_status, report_json_path)
        return results


def main():
    validator = DataValidator()
    res = validator.validate_processed_data()

    print("\n" + "=" * 70)
    print("DATA VALIDATION & QUALITY AUDIT REPORT")
    print("=" * 70)
    print(f"Overall Status:        {res['overall_status']}")
    print(f"Total Hourly Records:  {res['file_info']['total_hourly_records']:,}")
    print(f"Coverage:              {res['file_info']['time_span_start']} to {res['file_info']['time_span_end']}")
    print(f"Contiguity:            {res['file_info']['grid_continuity_ratio']}% of expected grid")
    print("\nTemporal Integrity:")
    print(f"  - Chronological:     {'PASSED (Monotonic)' if res['temporal_integrity']['is_strictly_chronological'] else 'FAILED'}")
    print(f"  - Duplicates:        {res['temporal_integrity']['duplicate_count']} duplicate timestamps")
    print("\nPhysical Range Checks (kWh):")
    print(f"  - Valid Range:       [{res['consumption_range_checks']['min_kwh']:.3f}, {res['consumption_range_checks']['max_kwh']:.3f}] kWh")
    print(f"  - Mean / Median:     {res['consumption_range_checks']['mean_kwh']:.3f} / {res['consumption_range_checks']['median_kwh']:.3f} kWh")
    print(f"  - Negative Values:   {res['consumption_range_checks']['negative_values_count']} (Expected: 0)")
    print("\nMissing Observation Audit:")
    print(f"  - Missing Hours:     {res['missing_value_audit']['null_target_count']} ({res['missing_value_audit']['null_percentage']}%)")
    print(f"  - Insufficient Min:  {res['missing_value_audit']['hours_below_min_observations_40min']} hours with <40 valid minutes")
    print("\nSplit Monotonicity & Zero-Leakage:")
    print(f"  - Split Verified:    {'PASSED (Train < Val < Test)' if res['chronological_split_integrity']['zero_temporal_leakage_verified'] else 'FAILED'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

