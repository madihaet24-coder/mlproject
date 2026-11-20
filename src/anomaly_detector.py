"""Adaptive Rolling Median + MAD anomaly detector and Fixed-Threshold baseline."""

from collections import deque
from typing import Deque, List, Optional

import numpy as np
import pandas as pd

from src.config import AnomalyDetectorConfig


class AdaptiveAnomalyDetector:
    """Residual-based adaptive anomaly detector with baseline protection and event grouping."""

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()

    @staticmethod
    def _compute_median_mad(values: np.ndarray, eps: float) -> tuple[float, float]:
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return med, max(mad, eps)

    def _effective_threshold(
        self,
        raw_mad_thresh: float,
        signed_res: float,
        actual_val: float,
        pred_val: float,
        med: float = 0.0,
        scale: float = 0.0,
    ) -> float:
        """Apply minimum residual error condition while preserving genuine sudden drops."""
        is_sudden_drop_candidate = (
            signed_res < 0.0 and pred_val >= 0.55 and actual_val <= 0.18 * pred_val
        )
        if is_sudden_drop_candidate:
            drop_k = max(1.8, self.config.sensitivity - 1.6)
            drop_mad_thresh = (med + drop_k * scale) if scale > 0.0 else raw_mad_thresh
            drop_floor = min(self.config.min_residual_kwh, 0.45)
            return max(min(raw_mad_thresh, drop_mad_thresh), drop_floor)
        return max(raw_mad_thresh, self.config.min_residual_kwh)

    @staticmethod
    def _assign_event_ids(is_anomaly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Group consecutive True flags into discrete event IDs and first-hour alert flags."""
        n = len(is_anomaly)
        event_ids = np.zeros(n, dtype=int)
        is_event_alert = np.zeros(n, dtype=bool)
        current_event = 0
        in_event = False

        for i in range(n):
            if bool(is_anomaly[i]):
                if not in_event:
                    current_event += 1
                    in_event = True
                    is_event_alert[i] = True
                event_ids[i] = current_event
            else:
                in_event = False

        return event_ids, is_event_alert

    @staticmethod
    def group_anomaly_events(results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate consecutive flagged hours into discrete anomaly event records."""
        columns = [
            "event_id",
            "start_timestamp",
            "end_timestamp",
            "duration_hours",
            "anomaly_type",
            "peak_actual",
            "mean_actual",
            "mean_predicted",
            "peak_residual",
            "threshold_at_peak",
            "peak_score",
        ]
        if results_df.empty or not results_df["is_anomaly"].any():
            return pd.DataFrame(columns=columns)

        df_work = results_df.copy()
        if "event_id" not in df_work.columns:
            ev_ids, _ = AdaptiveAnomalyDetector._assign_event_ids(
                df_work["is_anomaly"].to_numpy(dtype=bool)
            )
            df_work["event_id"] = ev_ids

        anoms = df_work[df_work["is_anomaly"] == True].copy()  # noqa: E712
        events: List[dict] = []

        for ev_id, grp in anoms.groupby("event_id", sort=True):
            peak_pos = grp["residual"].idxmax()
            # Choose dominant event type (Sustained Shift if duration >= 3 and positive, else peak's type)
            peak_type = str(grp.loc[peak_pos, "anomaly_type"])
            if len(grp) >= 3 and "Sustained Shift" in grp["anomaly_type"].values:
                ev_type = "Sustained Shift"
            else:
                ev_type = peak_type

            events.append(
                {
                    "event_id": int(ev_id),
                    "start_timestamp": grp["timestamp"].iloc[0],
                    "end_timestamp": grp["timestamp"].iloc[-1],
                    "duration_hours": int(len(grp)),
                    "anomaly_type": ev_type,
                    "peak_actual": round(float(grp["actual"].max()), 4),
                    "mean_actual": round(float(grp["actual"].mean()), 4),
                    "mean_predicted": round(float(grp["predicted"].mean()), 4),
                    "peak_residual": round(float(grp["residual"].max()), 4),
                    "threshold_at_peak": round(float(grp.loc[peak_pos, "threshold"]), 4),
                    "peak_score": round(float(grp["score"].max()), 2),
                }
            )

        return pd.DataFrame(events, columns=columns)

    def detect(
        self,
        actual,
        predicted,
        timestamps=None,
    ) -> pd.DataFrame:
        """Run online strictly past-only adaptive anomaly detection with baseline protection."""
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
        rebaseline_cap = max(1, self.config.rebaseline_hours)
        pending_regime: Deque[float] = deque(maxlen=rebaseline_cap)

        thresholds = np.zeros(n, dtype=float)
        scores = np.zeros(n, dtype=float)
        is_anomaly = np.zeros(n, dtype=bool)
        anomaly_types = ["Normal"] * n

        consecutive_violations = 0
        consecutive_positive = 0

        for i in range(n):
            r_t = float(abs_res[i])
            s_t = float(signed_res[i])
            a_t = float(actual_arr[i])
            p_t = float(pred_arr[i])

            if len(history) < self.config.min_history:
                if len(history) > 0:
                    med, mad = self._compute_median_mad(np.asarray(history), self.config.eps)
                    scale = 1.4826 * mad
                    raw_mad = med + self.config.sensitivity * scale
                    thresh_t = self._effective_threshold(raw_mad, s_t, a_t, p_t, med=med, scale=scale)
                    score_t = (r_t - med) / scale
                else:
                    init_scale = 1.4826 * max(r_t, self.config.eps)
                    raw_mad = r_t + self.config.sensitivity * init_scale
                    thresh_t = self._effective_threshold(
                        raw_mad, s_t, a_t, p_t, med=r_t, scale=init_scale
                    )
                    score_t = 0.0
                thresholds[i] = thresh_t
                scores[i] = score_t
                history.append(r_t)
                continue

            med, mad = self._compute_median_mad(np.asarray(history), self.config.eps)
            scale = 1.4826 * mad
            raw_mad = med + self.config.sensitivity * scale
            thresh_t = self._effective_threshold(raw_mad, s_t, a_t, p_t, med=med, scale=scale)
            score_t = (r_t - med) / scale

            raw_violation = r_t > thresh_t
            if raw_violation:
                consecutive_violations += 1
                pending_regime.append(r_t)
            else:
                consecutive_violations = 0
                consecutive_positive = 0
                pending_regime.clear()

            # Controlled re-baselining when sustained new consumption pattern persists > rebaseline_hours
            if (
                self.config.rebaseline_hours > 0
                and consecutive_violations > self.config.rebaseline_hours
            ):
                for val in pending_regime:
                    history.append(val)
                pending_regime.clear()
                consecutive_violations = 0
                consecutive_positive = 0

                med, mad = self._compute_median_mad(np.asarray(history), self.config.eps)
                scale = 1.4826 * mad
                raw_mad = med + self.config.sensitivity * scale
                thresh_t = self._effective_threshold(raw_mad, s_t, a_t, p_t, med=med, scale=scale)
                score_t = (r_t - med) / scale
                flagged = r_t > thresh_t
            else:
                is_severe_spike = r_t >= (self.config.severe_spike_multiplier * thresh_t)
                is_severe_drop = s_t < 0.0 and p_t >= 0.55 and a_t <= 0.05
                flagged = raw_violation and (
                    is_severe_spike
                    or is_severe_drop
                    or consecutive_violations >= self.config.persistence
                )

            thresholds[i] = thresh_t
            scores[i] = score_t
            is_anomaly[i] = flagged

            if flagged:
                if s_t < 0:
                    anomaly_types[i] = "Sudden Drop"
                    consecutive_positive = 0
                else:
                    consecutive_positive += 1
                    anomaly_types[i] = (
                        "Sustained Shift" if consecutive_positive >= 3 else "High Spike"
                    )

                if not self.config.exclude_anomalies:
                    history.append(r_t)
            else:
                # Preserve normal diurnal variance while preventing extreme spikes from contaminating H_t
                history.append(min(r_t, thresh_t))

        event_ids, is_event_alert = self._assign_event_ids(is_anomaly)

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
                "event_id": event_ids,
                "is_event_alert": is_event_alert,
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

        event_ids, is_event_alert = AdaptiveAnomalyDetector._assign_event_ids(is_anomaly)

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
                "event_id": event_ids,
                "is_event_alert": is_event_alert,
            }
        )
