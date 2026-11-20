"""Controlled synthetic anomaly injection, k-sensitivity selection, and benchmark evaluation."""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.anomaly_detector import AdaptiveAnomalyDetector
from src.config import AnomalyDetectorConfig, SyntheticScenarioConfig


class SyntheticAnomalyInjector:
    """Injects reproducible controlled anomaly scenarios into a test or validation stream."""

    def __init__(self, config: Optional[SyntheticScenarioConfig] = None):
        self.config = config or SyntheticScenarioConfig()

    def inject_scenarios(
        self,
        df: pd.DataFrame,
        target_col: str = "Global_active_power",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Inject temporary spikes, sustained shifts, repeated spikes, and sudden drops."""
        rng = np.random.default_rng(self.config.random_seed)
        df_out = df.copy().reset_index(drop=True)
        n = len(df_out)

        df_out["ground_truth_label"] = 0
        df_out["ground_truth_scenario"] = "normal"

        if n < 20:
            raise ValueError(f"Test stream too short ({n} rows) for synthetic scenario injection.")

        warmup = 200 if n >= 300 else (max(24, n // 5) if n >= 60 else max(4, n // 5))
        pad = 12 if n >= 300 else (2 if n >= 60 else 1)
        n_temp = self.config.n_temp_spikes if n >= 60 else 1
        n_rep = self.config.n_repeated_spikes if n >= 60 else 1
        events: List[Dict[str, Any]] = []
        event_id = 1
        occupied = np.zeros(n, dtype=bool)
        occupied[:warmup] = True

        def _find_free_slot(length: int) -> int:
            high = max(warmup + 1, n - length - pad)
            for _ in range(500):
                start = int(rng.integers(warmup, high))
                if not occupied[max(0, start - pad) : min(n, start + length + pad)].any():
                    occupied[start : start + length] = True
                    return start
            for start in range(warmup, max(warmup + 1, n - length)):
                if not occupied[start : start + length].any():
                    occupied[start : start + length] = True
                    return start
            raise RuntimeError("Could not find non-overlapping slot for synthetic scenario.")

        # 1. Temporary Spikes (2 hours each)
        for _ in range(n_temp):
            dur = 2
            s_idx = _find_free_slot(dur)
            spike_add = float(rng.uniform(3.5, 5.5))
            df_out.loc[s_idx : s_idx + dur - 1, target_col] += spike_add
            df_out.loc[s_idx : s_idx + dur - 1, "ground_truth_label"] = 1
            df_out.loc[s_idx : s_idx + dur - 1, "ground_truth_scenario"] = "temporary_spike"
            events.append(
                {
                    "event_id": event_id,
                    "scenario": "temporary_spike",
                    "start_idx": s_idx,
                    "end_idx": s_idx + dur - 1,
                    "duration_hours": dur,
                }
            )
            event_id += 1

        # 2. Sustained Level Shift (36 hours, scaled down for compact streams)
        dur_shift = (
            min(self.config.sustained_shift_hours, max(12, n // 15))
            if n >= 300
            else min(self.config.sustained_shift_hours, max(2, n // 15 if n >= 60 else n // 10))
        )
        s_idx = _find_free_slot(dur_shift)
        df_out.loc[s_idx : s_idx + dur_shift - 1, target_col] += 2.8
        df_out.loc[s_idx : s_idx + dur_shift - 1, "ground_truth_label"] = 1
        df_out.loc[s_idx : s_idx + dur_shift - 1, "ground_truth_scenario"] = "sustained_shift"
        events.append(
            {
                "event_id": event_id,
                "scenario": "sustained_shift",
                "start_idx": s_idx,
                "end_idx": s_idx + dur_shift - 1,
                "duration_hours": dur_shift,
            }
        )
        event_id += 1

        # 3. Repeated Spikes (periodic 2-hour bursts)
        for _ in range(n_rep):
            dur = 2
            s_idx = _find_free_slot(dur)
            df_out.loc[s_idx : s_idx + dur - 1, target_col] += float(rng.uniform(3.2, 4.8))
            df_out.loc[s_idx : s_idx + dur - 1, "ground_truth_label"] = 1
            df_out.loc[s_idx : s_idx + dur - 1, "ground_truth_scenario"] = "repeated_spikes"
            events.append(
                {
                    "event_id": event_id,
                    "scenario": "repeated_spikes",
                    "start_idx": s_idx,
                    "end_idx": s_idx + dur - 1,
                    "duration_hours": dur,
                }
            )
            event_id += 1

        # 4. Sudden Drops (near-zero outage during active consumption hours)
        dur_drop = min(
            self.config.sudden_drop_hours,
            12 if n >= 300 else (max(3, n // 20) if n >= 60 else max(2, n // 12)),
        )
        best_drop_start = None
        best_drop_score = -1.0
        drop_high = max(warmup + 1, n - dur_drop - pad)
        for _ in range(120):
            cand = int(rng.integers(warmup, drop_high))
            if occupied[max(0, cand - pad) : min(n, cand + dur_drop + pad)].any():
                continue
            window_vals = df_out.loc[cand : cand + dur_drop - 1, target_col]
            score_val = float(window_vals.min() + 0.25 * window_vals.mean())
            if score_val > best_drop_score:
                best_drop_score = score_val
                best_drop_start = cand
            if float(window_vals.min()) >= 0.75:
                break
        if best_drop_start is None:
            best_drop_start = _find_free_slot(dur_drop)
        else:
            occupied[best_drop_start : best_drop_start + dur_drop] = True
        s_idx = best_drop_start
        df_out.loc[s_idx : s_idx + dur_drop - 1, target_col] = 0.01
        df_out.loc[s_idx : s_idx + dur_drop - 1, "ground_truth_label"] = 1
        df_out.loc[s_idx : s_idx + dur_drop - 1, "ground_truth_scenario"] = "sudden_drop"
        events.append(
            {
                "event_id": event_id,
                "scenario": "sudden_drop",
                "start_idx": s_idx,
                "end_idx": s_idx + dur_drop - 1,
                "duration_hours": dur_drop,
            }
        )

        events_df = pd.DataFrame(events)
        return df_out, events_df

    @staticmethod
    def inject_routine_change(
        df: pd.DataFrame,
        target_col: str = "Global_active_power",
        start_idx: int = 220,
        duration_hours: int = 168,
        shift_kwh: float = 1.4,
    ) -> pd.DataFrame:
        """Inject a sustained normal routine shift to verify controlled re-baselining."""
        df_out = df.copy().reset_index(drop=True)
        end_idx = min(len(df_out), start_idx + duration_hours)
        df_out.loc[start_idx : end_idx - 1, target_col] += shift_kwh
        return df_out


class Evaluator:
    """Computes point-wise, controlled-window, event-level, and before-vs-after metrics."""

    @staticmethod
    def _classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, n_hours: int
    ) -> Dict[str, Any]:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        days = max(1e-6, n_hours / 24.0)
        fa_per_day = float(fp / days)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_alarms_per_day": round(fa_per_day, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    @staticmethod
    def summarize_unlabelled_stream(results_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize flagged hours and grouped anomaly events on an unlabelled stream."""
        total_hours = int(len(results_df))
        flagged_hours = int(results_df["is_anomaly"].sum())
        grouped_events = AdaptiveAnomalyDetector.group_anomaly_events(results_df)
        grouped_count = int(len(grouped_events))
        days = max(1e-6, total_hours / 24.0)
        return {
            "total_hours": total_hours,
            "flagged_hours": flagged_hours,
            "flagged_rate_percent": round(100.0 * flagged_hours / max(1, total_hours), 2),
            "grouped_events": grouped_count,
            "events_per_day": round(grouped_count / days, 4),
        }

    @staticmethod
    def evaluate_scenarios(
        results_df: pd.DataFrame, events_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate detector outputs against ground truth labels and injected events."""
        y_true = results_df["ground_truth_label"].to_numpy(dtype=int)
        y_pred = results_df["is_anomaly"].to_numpy(dtype=int)
        n_hours = len(results_df)

        overall = Evaluator._classification_metrics(y_true, y_pred, n_hours)

        # Event-level evaluation
        detected_events = 0
        delays: List[float] = []
        controlled_mask = np.zeros(n_hours, dtype=bool)

        for _, ev in events_df.iterrows():
            s_idx = int(ev["start_idx"])
            e_idx = int(ev["end_idx"])
            win_start = max(0, s_idx - 12)
            win_end = min(n_hours - 1, e_idx + 12)
            controlled_mask[win_start : win_end + 1] = True

            ev_preds = y_pred[s_idx : e_idx + 1]
            hit_positions = np.where(ev_preds == 1)[0]
            if len(hit_positions) > 0:
                detected_events += 1
                delays.append(float(hit_positions[0]))

        total_events = len(events_df)
        event_rate = float(detected_events / total_events) if total_events > 0 else 0.0
        avg_delay = float(np.mean(delays)) if delays else 0.0

        controlled = Evaluator._classification_metrics(
            y_true[controlled_mask],
            y_pred[controlled_mask],
            int(controlled_mask.sum()),
        )

        # Per-scenario breakdown
        per_scenario: Dict[str, Dict[str, Any]] = {}
        scenarios = [s for s in results_df["ground_truth_scenario"].unique() if s != "normal"]
        for sc in scenarios:
            sc_mask = (results_df["ground_truth_scenario"] == sc).to_numpy()
            anomalous_samples = int(sc_mask.sum())
            detected_samples = int((sc_mask & (y_pred == 1)).sum())
            sc_recall = float(detected_samples / anomalous_samples) if anomalous_samples > 0 else 0.0
            per_scenario[sc] = {
                "anomalous_samples": anomalous_samples,
                "detected_samples": detected_samples,
                "recall": round(sc_recall, 4),
            }

        flagged_hours = int(y_pred.sum())
        grouped_count = int(len(AdaptiveAnomalyDetector.group_anomaly_events(results_df)))

        return {
            "overall": overall,
            "controlled_benchmark": controlled,
            "event_level": {
                "total_events": total_events,
                "detected_events": detected_events,
                "event_detection_rate": round(event_rate, 4),
                "average_detection_delay_hours": round(avg_delay, 2),
            },
            "per_scenario": per_scenario,
            "stream_counts": {
                "total_flagged_hours": flagged_hours,
                "flagged_rate_percent": round(100.0 * flagged_hours / max(1, n_hours), 2),
                "grouped_event_count": grouped_count,
            },
        }

    @staticmethod
    def _evaluate_detector_on_split(
        config: AnomalyDetectorConfig,
        actual: np.ndarray,
        predicted: np.ndarray,
        timestamps: pd.Series,
        scenario_seed: int,
    ) -> Dict[str, Any]:
        """Run detector on both real unlabelled stream and controlled synthetic benchmark."""
        detector = AdaptiveAnomalyDetector(config)
        real_res = detector.detect(actual=actual, predicted=predicted, timestamps=timestamps)
        real_summary = Evaluator.summarize_unlabelled_stream(real_res)

        df_stream = pd.DataFrame({"Datetime": pd.to_datetime(timestamps).values, "Global_active_power": actual})
        injector = SyntheticAnomalyInjector(SyntheticScenarioConfig(random_seed=scenario_seed))
        df_inj, events_df = injector.inject_scenarios(df_stream, "Global_active_power")

        syn_res = detector.detect(
            actual=df_inj["Global_active_power"].to_numpy(dtype=float),
            predicted=predicted,
            timestamps=timestamps,
        )
        syn_res["ground_truth_label"] = df_inj["ground_truth_label"].values
        syn_res["ground_truth_scenario"] = df_inj["ground_truth_scenario"].values
        syn_eval = Evaluator.evaluate_scenarios(syn_res, events_df)

        return {
            "real_unlabelled_stream": real_summary,
            "synthetic_benchmark": syn_eval,
        }

    @staticmethod
    def run_k_sensitivity_sweep(
        actual_val: np.ndarray,
        pred_val: np.ndarray,
        ts_val: pd.Series,
        actual_test: np.ndarray,
        pred_test: np.ndarray,
        ts_test: pd.Series,
        base_config: Optional[AnomalyDetectorConfig] = None,
    ) -> Dict[str, Any]:
        """Evaluate candidate k values (3.0, 3.5, 4.0, 4.5) and select optimal k on validation data."""
        cfg = base_config or AnomalyDetectorConfig()
        k_rows: List[Dict[str, Any]] = []

        for k_val in cfg.candidate_k_values:
            k_cfg = replace(cfg, sensitivity=float(k_val))
            val_out = Evaluator._evaluate_detector_on_split(
                k_cfg, actual_val, pred_val, ts_val, scenario_seed=101
            )
            test_out = Evaluator._evaluate_detector_on_split(
                k_cfg, actual_test, pred_test, ts_test, scenario_seed=42
            )

            val_real = val_out["real_unlabelled_stream"]
            val_syn = val_out["synthetic_benchmark"]
            test_real = test_out["real_unlabelled_stream"]
            test_syn = test_out["synthetic_benchmark"]

            # Balanced validation selection score: penalize excess alarms & over-tightening k > 4.0
            val_ctrl_f1 = val_syn["controlled_benchmark"]["f1_score"]
            val_ev_rate = val_syn["event_level"]["event_detection_rate"]
            min_sc_recall = min(
                (sc["recall"] for sc in val_syn["per_scenario"].values()),
                default=0.0,
            )
            excess_rate_penalty = 0.05 * max(0.0, val_real["flagged_rate_percent"] - 1.45)
            over_tighten_penalty = 0.04 * max(0.0, float(k_val) - 4.0)
            selection_score = round(
                0.60 * val_ctrl_f1
                + 0.40 * min_sc_recall
                - excess_rate_penalty
                - over_tighten_penalty
                + (0.20 if val_ev_rate >= 0.95 else -0.50),
                4,
            )

            k_rows.append(
                {
                    "k": float(k_val),
                    "validation": {
                        "flagged_hours": val_real["flagged_hours"],
                        "flagged_rate_percent": val_real["flagged_rate_percent"],
                        "grouped_events": val_real["grouped_events"],
                        "synthetic_precision": val_syn["controlled_benchmark"]["precision"],
                        "synthetic_recall": val_syn["controlled_benchmark"]["recall"],
                        "synthetic_f1": val_ctrl_f1,
                        "synthetic_event_rate": val_ev_rate,
                        "synthetic_delay_hours": val_syn["event_level"]["average_detection_delay_hours"],
                        "selection_score": selection_score,
                    },
                    "test": {
                        "flagged_hours": test_real["flagged_hours"],
                        "flagged_rate_percent": test_real["flagged_rate_percent"],
                        "grouped_events": test_real["grouped_events"],
                        "synthetic_precision": test_syn["controlled_benchmark"]["precision"],
                        "synthetic_recall": test_syn["controlled_benchmark"]["recall"],
                        "synthetic_f1": test_syn["controlled_benchmark"]["f1_score"],
                        "synthetic_event_rate": test_syn["event_level"]["event_detection_rate"],
                        "synthetic_delay_hours": test_syn["event_level"]["average_detection_delay_hours"],
                    },
                }
            )

        # Select best k from validation metrics without blindly picking largest k
        eligible = [r for r in k_rows if r["validation"]["synthetic_event_rate"] >= 0.90]
        pool = eligible if eligible else k_rows
        best_row = max(pool, key=lambda r: r["validation"]["selection_score"])

        return {
            "candidate_values": list(cfg.candidate_k_values),
            "selected_k": best_row["k"],
            "selection_criterion": (
                "Maximizes validation controlled F1 minus false-alarm rate penalty "
                "subject to synthetic event detection rate >= 90%."
            ),
            "results": k_rows,
        }

    @staticmethod
    def compare_before_after(
        actual_val: np.ndarray,
        pred_val: np.ndarray,
        ts_val: pd.Series,
        actual_test: np.ndarray,
        pred_test: np.ndarray,
        ts_test: pd.Series,
        improved_config: AnomalyDetectorConfig,
    ) -> Dict[str, Any]:
        """Compare Legacy (Before) vs Improved (After) detectors on Validation and Test splits."""
        legacy_cfg = AnomalyDetectorConfig.legacy_config()

        def _build_split_comp(actual, pred, ts, seed):
            before = Evaluator._evaluate_detector_on_split(legacy_cfg, actual, pred, ts, seed)
            after = Evaluator._evaluate_detector_on_split(improved_config, actual, pred, ts, seed)

            b_hrs = before["real_unlabelled_stream"]["flagged_hours"]
            a_hrs = after["real_unlabelled_stream"]["flagged_hours"]
            b_ev = before["real_unlabelled_stream"]["grouped_events"]
            a_ev = after["real_unlabelled_stream"]["grouped_events"]

            return {
                "legacy_before": before,
                "improved_after": after,
                "reduction_summary": {
                    "flagged_hours_before": b_hrs,
                    "flagged_hours_after": a_hrs,
                    "flagged_hours_reduction_percent": round(100.0 * (b_hrs - a_hrs) / max(1, b_hrs), 1),
                    "grouped_events_before": b_ev,
                    "grouped_events_after": a_ev,
                    "grouped_events_reduction_percent": round(100.0 * (b_ev - a_ev) / max(1, b_ev), 1),
                },
            }

        return {
            "validation": _build_split_comp(actual_val, pred_val, ts_val, seed=101),
            "test": _build_split_comp(actual_test, pred_test, ts_test, seed=42),
        }
