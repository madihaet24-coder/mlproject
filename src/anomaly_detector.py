"""Adaptive Rolling Median + MAD anomaly detector and Fixed-Threshold baseline."""

from collections import deque
from typing import Deque, Optional

import numpy as np
import pandas as pd

from src.config import AnomalyDetectorConfig


class AdaptiveAnomalyDetector:
    """Residual-based adaptive anomaly detector with anomaly-exclusion policy."""

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()

    @staticmethod
    def _compute_median_mad(values: np.ndarray, eps: float) -> tuple[float, float]:
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return med, max(mad, eps)

    def detect(
        self,
        actual,
        predicted,
        timestamps=None,
    ) -> pd.DataFrame:
        """Run online strictly past-only adaptive anomaly detection."""
        actual_arr = np.asarray(actual, dtype=float)
        pred_arr = np.asarray(predicted, dtype=float)
        n = len(actual_arr)

        if timestamps is None:
            ts_arr = pd.date_range("2000-01-01", periods=n, freq="1h")
        else:
            ts_arr = pd.to_datetime(timestamps)

        signed_res = actual_arr - pred_arr
        abs_res = np.abs(signed_res)

        history: Deque[float] = deque(maxlen=self.config.window_size)
        thresholds = np.zeros(n, dtype=float)
        scores = np.zeros(n, dtype=float)
        is_anomaly = np.zeros(n, dtype=bool)
        anomaly_types = ["Normal"] * n

        consecutive_violations = 0
        consecutive_positive = 0

        for i in range(n):
            r_t = float(abs_res[i])
            s_t = float(signed_res[i])

            if len(history) < self.config.min_history:
                # Warm-up phase: build initial baseline without flagging
                if len(history) > 0:
                    med, mad = self._compute_median_mad(np.asarray(history), self.config.eps)
                    thresh_t = med + self.config.sensitivity * 1.4826 * mad
                    score_t = (r_t - med) / (1.4826 * mad)
                else:
                    thresh_t = np.nan
                    score_t = 0.0
                thresholds[i] = thresh_t
                scores[i] = score_t
                history.append(r_t)
                continue

            med, mad = self._compute_median_mad(np.asarray(history), self.config.eps)
            scale = 1.4826 * mad
            thresh_t = med + self.config.sensitivity * scale
            score_t = (r_t - med) / scale

            thresholds[i] = thresh_t
            scores[i] = score_t

            raw_violation = r_t > thresh_t
            if raw_violation:
                consecutive_violations += 1
            else:
                consecutive_violations = 0
                consecutive_positive = 0

            flagged = raw_violation and (consecutive_violations >= self.config.persistence)
            is_anomaly[i] = flagged

            if flagged:
                if s_t < 0:
                    anomaly_types[i] = "Sudden Drop"
                    consecutive_positive = 0
                else:
                    consecutive_positive += 1
                    anomaly_types[i] = "Sustained Shift" if consecutive_positive >= 3 else "High Spike"

                if not self.config.exclude_anomalies:
                    history.append(r_t)
            else:
                history.append(r_t)

        return pd.DataFrame(
            {
                "timestamp": ts_arr,
                "actual": actual_arr,
                "predicted": pred_arr,
                "residual": abs_res,
                "signed_residual": signed_res,
                "threshold": thresholds,
                "score": scores,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_types,
            }
        )


class FixedThresholdBaseline:
    """Static 3-sigma residual threshold baseline calibrated on validation residuals."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold: Optional[float] = threshold

    def fit_from_residuals(self, residuals, multiplier: float = 3.0) -> float:
        """Calibrate a static threshold from validation residuals."""
        res_arr = np.abs(np.asarray(residuals, dtype=float))
        res_arr = res_arr[~np.isnan(res_arr)]
        if len(res_arr) == 0:
            raise ValueError("Cannot calibrate FixedThresholdBaseline on empty residuals.")
        mean_r = float(np.mean(res_arr))
        std_r = float(np.std(res_arr))
        self.threshold = max(mean_r + multiplier * std_r, 1e-4)
        return self.threshold

    def detect(self, actual, predicted, timestamps=None) -> pd.DataFrame:
        """Detect anomalies using the calibrated static threshold."""
        if self.threshold is None:
            raise RuntimeError("FixedThresholdBaseline must be fitted before calling detect().")

        actual_arr = np.asarray(actual, dtype=float)
        pred_arr = np.asarray(predicted, dtype=float)
        n = len(actual_arr)

        if timestamps is None:
            ts_arr = pd.date_range("2000-01-01", periods=n, freq="1h")
        else:
            ts_arr = pd.to_datetime(timestamps)

        signed_res = actual_arr - pred_arr
        abs_res = np.abs(signed_res)
        is_anomaly = abs_res > self.threshold
        scores = abs_res / self.threshold

        anomaly_types = []
        for flag, s_val in zip(is_anomaly, signed_res):
            if not flag:
                anomaly_types.append("Normal")
            elif s_val < 0:
                anomaly_types.append("Sudden Drop")
            else:
                anomaly_types.append("High Spike")

        return pd.DataFrame(
            {
                "timestamp": ts_arr,
                "actual": actual_arr,
                "predicted": pred_arr,
                "residual": abs_res,
                "signed_residual": signed_res,
                "threshold": np.full(n, self.threshold, dtype=float),
                "score": scores,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_types,
            }
        )
