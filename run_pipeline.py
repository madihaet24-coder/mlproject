"""Command-line interface to execute the full anomaly detection pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --force-download --force-train
"""

import argparse
import logging
import sys
from pathlib import Path

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
    args = parser.parse_args()

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
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error("Pipeline run failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

