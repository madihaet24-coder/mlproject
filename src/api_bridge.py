"""Lightweight JSON bridge for the Node.js Express backend."""

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src.anomaly_detector import AdaptiveAnomalyDetector
from src.config import REPORTS_DIR, AnomalyDetectorConfig, DataConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.preprocessing import Preprocessor


def _load_test_predictions():
    anom_csv = REPORTS_DIR / "detected_anomalies.csv"
    if anom_csv.exists():
        df_cached = pd.read_csv(anom_csv, usecols=["timestamp", "actual", "predicted"])
        if not df_cached.empty:
            return (
                df_cached["actual"].to_numpy(dtype=float),
                df_cached["predicted"].to_numpy(dtype=float),
                pd.to_datetime(df_cached["timestamp"]),
            )

    loader = DataLoader()
    df_hourly = loader.process_and_save()
    prep = Preprocessor()
    df_clean = prep.clean_hourly_data(df_hourly)
    _, df_val, _ = prep.split_chronological(df_clean)

    engineer = FeatureEngineer()
    df_feats = engineer.create_features(df_clean)
    val_max_dt = df_val[DataConfig.datetime_col].max()
    feats_test = df_feats[df_feats[DataConfig.datetime_col] > val_max_dt]
    X_test, y_test, ts_test = engineer.get_feature_matrix(feats_test)

    trainer = ModelTrainer()
    trainer.load_model("xgboost_forecaster")
    preds_test = trainer.predict(X_test)
    return y_test.to_numpy(dtype=float), preds_test, ts_test


def _serialize_rows(df_subset: pd.DataFrame) -> list[dict]:
    return [
        {
            "timestamp": pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "actual": round(float(r["actual"]), 4),
            "predicted": round(float(r["predicted"]), 4),
            "residual": round(float(r["residual"]), 4),
            "threshold": round(float(r["threshold"]), 4) if np.isfinite(r["threshold"]) else 0.0,
            "score": round(float(r["score"]), 3) if np.isfinite(r["score"]) else 0.0,
            "is_anomaly": bool(r["is_anomaly"]),
            "anomaly_type": str(r["anomaly_type"]),
            "event_id": int(r["event_id"]),
            "is_event_alert": bool(r["is_event_alert"]),
        }
        for _, r in df_subset.iterrows()
    ]


def cmd_detect(k_val: float, window_size: int, limit: int) -> None:
    y_test, preds_test, ts_test = _load_test_predictions()
    cfg = replace(
        AnomalyDetectorConfig(),
        sensitivity=float(k_val),
        window_size=int(window_size),
    )
    detector = AdaptiveAnomalyDetector(cfg)
    res_df = detector.detect(actual=y_test, predicted=preds_test, timestamps=ts_test)
    events_df = AdaptiveAnomalyDetector.group_anomaly_events(res_df)

    # Legacy baseline counts for inline comparison
    legacy_res = AdaptiveAnomalyDetector(AnomalyDetectorConfig.legacy_config()).detect(
        actual=y_test, predicted=preds_test, timestamps=ts_test
    )
    legacy_events = AdaptiveAnomalyDetector.group_anomaly_events(legacy_res)

    if limit > 0:
        sliced = res_df.iloc[-limit:]
    elif len(res_df) > 1200:
        step = max(1, len(res_df) // 1000)
        keep_mask = np.zeros(len(res_df), dtype=bool)
        keep_mask[::step] = True
        keep_mask |= res_df["is_anomaly"].to_numpy(dtype=bool)
        keep_mask[-1] = True
        sliced = res_df.iloc[keep_mask]
    else:
        sliced = res_df

    rows = _serialize_rows(sliced)
    flagged_rows = _serialize_rows(res_df[res_df["is_anomaly"] == True])  # noqa: E712

    events = [
        {
            "event_id": int(r["event_id"]),
            "start_timestamp": pd.to_datetime(r["start_timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "end_timestamp": pd.to_datetime(r["end_timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hours": int(r["duration_hours"]),
            "anomaly_type": str(r["anomaly_type"]),
            "peak_actual": float(r["peak_actual"]),
            "mean_actual": float(r["mean_actual"]),
            "mean_predicted": float(r["mean_predicted"]),
            "peak_residual": float(r["peak_residual"]),
            "threshold_at_peak": float(r["threshold_at_peak"]),
            "peak_score": float(r["peak_score"]),
        }
        for _, r in events_df.iterrows()
    ]

    total = len(res_df)
    flagged = int(res_df["is_anomaly"].sum())
    print(
        json.dumps(
            {
                "k": float(k_val),
                "window_size": int(window_size),
                "min_residual_kwh": cfg.min_residual_kwh,
                "persistence": cfg.persistence,
                "severe_spike_multiplier": cfg.severe_spike_multiplier,
                "rebaseline_hours": cfg.rebaseline_hours,
                "total": total,
                "anomalyCount": flagged,
                "ratePercent": round(100.0 * flagged / max(1, total), 2),
                "groupedEventsCount": len(events),
                "beforeFlaggedHours": int(legacy_res["is_anomaly"].sum()),
                "beforeEventsCount": int(len(legacy_events)),
                "events": events,
                "flaggedRows": flagged_rows,
                "rows": rows,
            }
        )
    )


def cmd_simulate(start_idx: int, horizon: int) -> None:
    loader = DataLoader()
    df_hourly = loader.process_and_save()
    prep = Preprocessor()
    df_clean = prep.clean_hourly_data(df_hourly)

    trainer = ModelTrainer()
    trainer.load_model("xgboost_forecaster")

    sim_df = FeatureEngineer.forecast_recursive_24h(
        df_history=df_clean,
        model_trainer=trainer,
        start_idx=start_idx,
        horizon_hours=horizon,
    )
    records = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "actual_kwh": round(float(row["actual_kwh"]), 4),
            "forecast_kwh": round(float(row["forecast_kwh"]), 4),
        }
        for _, row in sim_df.iterrows()
    ]
    print(json.dumps({"horizon": horizon, "start_idx": start_idx, "series": records}))


def cmd_profiles() -> None:
    loader = DataLoader()
    df_raw = loader.process_and_save()
    df = Preprocessor().clean_hourly_data(df_raw)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["hour"] = df["Datetime"].dt.hour
    df["dow"] = df["Datetime"].dt.day_name()

    diurnal = (
        df.groupby("hour")["Global_active_power"]
        .agg(["mean", "median"])
        .reindex(range(24))
        .fillna(0.0)
        .reset_index()
    )
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = (
        df.groupby("dow")["Global_active_power"]
        .mean()
        .reindex(dow_order)
        .fillna(0.0)
        .reset_index()
    )

    out = {
        "total_rows": int(len(df)),
        "start_date": str(df["Datetime"].min()),
        "end_date": str(df["Datetime"].max()),
        "diurnal": [
            {"hour": int(r["hour"]), "mean": round(float(r["mean"]), 4), "median": round(float(r["median"]), 4)}
            for _, r in diurnal.iterrows()
        ],
        "weekly": [
            {"day": str(r["dow"])[:3], "mean": round(float(r["Global_active_power"]), 4)}
            for _, r in weekly.iterrows()
        ],
    }
    print(json.dumps(out))


def cmd_datasets() -> None:
    loader = DataLoader()
    print(json.dumps(loader.list_datasets()))


def cmd_ingest(
    file_path: str,
    name: str | None,
    datetime_col: str | None,
    target_col: str | None,
) -> None:
    loader = DataLoader()
    entry = loader.ingest_custom_dataset(
        source_path=file_path,
        dataset_name=name,
        datetime_col=datetime_col or None,
        target_col=target_col or None,
        activate=True,
    )
    print(json.dumps({"ingested": entry, "registry": loader.list_datasets()}))


def cmd_activate(dataset_id: str) -> None:
    loader = DataLoader()
    registry = loader.activate_dataset(dataset_id)
    print(json.dumps({"registry": registry}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim_p = sub.add_parser("simulate")
    sim_p.add_argument("--start-idx", type=int, required=True)
    sim_p.add_argument("--horizon", type=int, default=24)

    det_p = sub.add_parser("detect")
    det_p.add_argument("--k", type=float, default=4.0)
    det_p.add_argument("--window", type=int, default=168)
    det_p.add_argument("--limit", type=int, default=500)

    sub.add_parser("profiles")
    sub.add_parser("datasets")

    ing_p = sub.add_parser("ingest")
    ing_p.add_argument("--file", required=True)
    ing_p.add_argument("--name", default=None)
    ing_p.add_argument("--datetime-col", default=None)
    ing_p.add_argument("--target-col", default=None)

    act_p = sub.add_parser("activate")
    act_p.add_argument("--id", required=True)

    args = parser.parse_args()
    if args.cmd == "simulate":
        cmd_simulate(args.start_idx, args.horizon)
    elif args.cmd == "detect":
        cmd_detect(args.k, args.window, args.limit)
    elif args.cmd == "profiles":
        cmd_profiles()
    elif args.cmd == "datasets":
        cmd_datasets()
    elif args.cmd == "ingest":
        cmd_ingest(args.file, args.name, args.datetime_col, args.target_col)
    elif args.cmd == "activate":
        cmd_activate(args.id)


if __name__ == "__main__":
    main()
