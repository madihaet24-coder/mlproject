"""Lightweight JSON bridge for the Node.js Express backend."""

import argparse
import json
import sys

import pandas as pd

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.preprocessing import Preprocessor


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
    df = pd.read_csv(loader.processed_csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["hour"] = df["Datetime"].dt.hour
    df["dow"] = df["Datetime"].dt.day_name()

    diurnal = (
        df.groupby("hour")["Global_active_power"]
        .agg(["mean", "median"])
        .reset_index()
    )
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = (
        df.groupby("dow")["Global_active_power"]
        .mean()
        .reindex(dow_order)
        .reset_index()
    )

    out = {
        "total_rows": int(len(df)),
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


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim_p = sub.add_parser("simulate")
    sim_p.add_argument("--start-idx", type=int, required=True)
    sim_p.add_argument("--horizon", type=int, default=24)

    sub.add_parser("profiles")

    args = parser.parse_args()
    if args.cmd == "simulate":
        cmd_simulate(args.start_idx, args.horizon)
    elif args.cmd == "profiles":
        cmd_profiles()


if __name__ == "__main__":
    main()
