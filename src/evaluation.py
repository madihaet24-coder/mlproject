"""Controlled synthetic anomaly injection and benchmark evaluation metrics."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import SyntheticScenarioConfig


class SyntheticAnomalyInjector:
    """Injects reproducible controlled anomaly scenarios into a test stream."""

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

        if n < 300:
            raise ValueError(f"Test stream too short ({n} rows) for synthetic scenario injection.")

        events: List[Dict[str, Any]] = []
        event_id = 1
        occupied = np.zeros(n, dtype=bool)
        # Keep initial 200 hours clean for detector warm-up
        occupied[:200] = True

        def _find_free_slot(length: int) -> int:
            for _ in range(500):
                start = int(rng.integers(200, max(201, n - length - 24)))
                if not occupied[max(0, start - 12) : min(n, start + length + 12)].any():
                    occupied[start : start + length] = True
                    return start
            raise RuntimeError("Could not find non-overlapping slot for synthetic scenario.")

        # 1. Temporary Spikes (2 hours each)
        for _ in range(self.config.n_temp_spikes):
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

        # 2. Sustained Level Shift (36 hours)
        dur_shift = min(self.config.sustained_shift_hours, max(12, n // 15))
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
        for _ in range(self.config.n_repeated_spikes):
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

        # 4. Sudden Drops (near-zero outage)
        dur_drop = min(self.config.sudden_drop_hours, 12)
        s_idx = _find_free_slot(dur_drop)
        # Ensure high enough baseline or drop relative to local level
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


class Evaluator:
    """Computes point-wise, controlled-window, event-level, and per-scenario metrics."""

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_hours: int) -> Dict[str, Any]:
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
            # Include +/- 12h neighborhood for controlled window precision
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
        }
