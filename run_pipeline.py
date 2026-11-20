"""Command-line interface to execute the full anomaly detection pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --force-download --force-train
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data_loader import DataLoader
from src.pipeline import AnomalyDetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Household Electricity Anomaly Detection Pipeline")
    parser.add_argument("--force-download", action="store_true", help="Force redownload of the UCI dataset")
    parser.add_argument("--force-train", action="store_true", help="Force retraining of the forecasting model")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a custom CSV/TXT dataset to ingest, retrain, and evaluate")
    parser.add_argument("--dataset-name", type=str, default=None, help="Display name for the custom dataset")
    parser.add_argument("--datetime-col", type=str, default=None, help="Timestamp column name in custom dataset")
    parser.add_argument("--target-col", type=str, default=None, help="Consumption column name in custom dataset")
    parser.add_argument("--reset-default-dataset", action="store_true", help="Switch back to the default UCI dataset and retrain")
    args = parser.parse_args()

    loader = DataLoader()
    if args.reset_default_dataset:
        loader.activate_dataset("uci_default")
        args.force_train = True
    elif args.dataset:
        loader.ingest_custom_dataset(
            source_path=Path(args.dataset),
            dataset_name=args.dataset_name,
            datetime_col=args.datetime_col,
            target_col=args.target_col,
            activate=True,
        )
        args.force_train = True

    logger.info("Initializing Anomaly Detection Research Pipeline...")
    pipeline = AnomalyDetectionPipeline()

    try:
        results = pipeline.run_pipeline(
            force_recompute_data=args.force_download,
            force_retrain_model=args.force_train,
        )
        report = results["comparison_report"]
        adaptive_metrics = report["adaptive_detector"]["overall"]
        baseline_metrics = report["fixed_baseline"]["overall"]
        forecasting = report["forecasting_metrics"]

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Forecasting Model (XGBoost):")
        print(f"  - Test MAE:   {forecasting['mae']:.4f} kWh")
        print(f"  - Test RMSE:  {forecasting['rmse']:.4f} kWh")
        print(f"  - Test R^2:   {forecasting['r2']:.4f}")
        print("\nAnomaly Detection Performance Comparison:")
        print(f"  {'Metric':<28} | {'Adaptive MAD Detector':<22} | {'Fixed Baseline':<16}")
        print(f"  {'-'*28} | {'-'*22} | {'-'*16}")
        print(f"  {'Precision':<28} | {adaptive_metrics['precision']*100:>20.1f}% | {baseline_metrics['precision']*100:>14.1f}%")
        print(f"  {'Recall':<28} | {adaptive_metrics['recall']*100:>20.1f}% | {baseline_metrics['recall']*100:>14.1f}%")
        print(f"  {'F1-Score':<28} | {adaptive_metrics['f1_score']*100:>20.1f}% | {baseline_metrics['f1_score']*100:>14.1f}%")
        print(f"  {'False Alarms / Day':<28} | {adaptive_metrics['false_alarms_per_day']:>21.2f} | {baseline_metrics['false_alarms_per_day']:>15.2f}")
        print(f"  {'True Positives':<28} | {adaptive_metrics['true_positives']:>21} | {baseline_metrics['true_positives']:>15}")
        print(f"  {'False Positives':<28} | {adaptive_metrics['false_positives']:>21} | {baseline_metrics['false_positives']:>15}")

        if "k_sensitivity_sweep" in report:
            sweep = report["k_sensitivity_sweep"]
            print(f"\nValidation Sensitivity Sweep (Selected k = {sweep['selected_k']}):")
            print(f"  {'k':<6} | {'Val Flagged Hrs':<15} | {'Test Flagged Hrs':<16} | {'Val Ctrl F1':<11} | {'Val Ev Rate':<11}")
            print(f"  {'-'*6} | {'-'*15} | {'-'*16} | {'-'*11} | {'-'*11}")
            for row in sweep["results"]:
                v, t = row["validation"], row["test"]
                print(
                    f"  {row['k']:<6.1f} | {v['flagged_hours']:<15} | {t['flagged_hours']:<16} | "
                    f"{v['synthetic_f1']*100:>9.1f}% | {v['synthetic_event_rate']*100:>9.1f}%"
                )

        if "before_vs_after" in report:
            t_red = report["before_vs_after"]["test"]["reduction_summary"]
            v_red = report["before_vs_after"]["validation"]["reduction_summary"]
            print("\nBefore (Legacy k=3.0) vs After (Improved) False Alarm Reduction:")
            print(
                f"  - Validation Flagged Hours: {v_red['flagged_hours_before']} -> {v_red['flagged_hours_after']} "
                f"(-{v_red['flagged_hours_reduction_percent']:.1f}%), Events: {v_red['grouped_events_before']} -> {v_red['grouped_events_after']}"
            )
            print(
                f"  - Test Flagged Hours:       {t_red['flagged_hours_before']} -> {t_red['flagged_hours_after']} "
                f"(-{t_red['flagged_hours_reduction_percent']:.1f}%), Events: {t_red['grouped_events_before']} -> {t_red['grouped_events_after']}"
            )
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error("Pipeline run failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

