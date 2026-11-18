"""Unit tests for src.anomaly_detector."""

import numpy as np
from src.anomaly_detector import AdaptiveAnomalyDetector, FixedThresholdBaseline
from src.config import AnomalyDetectorConfig


def test_adaptive_detector_no_future_leakage():
    rng = np.random.default_rng(7)
    actual = 1.5 + rng.normal(0, 0.1, size=300)
    predicted = np.full(300, 1.5)

    actual_future_perturbed = actual.copy()
    actual_future_perturbed[200:] += 10.0

    det = AdaptiveAnomalyDetector(AnomalyDetectorConfig(sensitivity=3.0, window_size=48))
    res1 = det.detect(actual, predicted)
    res2 = det.detect(actual_future_perturbed, predicted)

    np.testing.assert_allclose(
        res1["threshold"].iloc[:200].to_numpy(),
        res2["threshold"].iloc[:200].to_numpy(),
        equal_nan=True,
    )


def test_zero_mad_numerical_safeguard():
    actual = np.ones(100)
    predicted = np.ones(100)
    det = AdaptiveAnomalyDetector(AnomalyDetectorConfig(eps=1e-4, window_size=24))
    res = det.detect(actual, predicted)
    assert np.isfinite(res["score"]).all()
    assert not res["is_anomaly"].any()


def test_anomaly_exclusion_prevents_baseline_contamination():
    actual = np.ones(120)
    predicted = np.ones(120)
    # Inject a huge spike at index 60
    actual[60] = 20.0

    det_exclude = AdaptiveAnomalyDetector(
        AnomalyDetectorConfig(window_size=30, min_history=24, exclude_anomalies=True)
    )
    det_include = AdaptiveAnomalyDetector(
        AnomalyDetectorConfig(window_size=30, min_history=24, exclude_anomalies=False)
    )
    res_ex = det_exclude.detect(actual, predicted)
    res_in = det_include.detect(actual, predicted)

    assert res_ex.loc[60, "is_anomaly"]
    # Excluding the spike keeps subsequent threshold <= threshold when included
    assert res_ex.loc[61, "threshold"] <= res_in.loc[61, "threshold"]


def test_fixed_threshold_baseline_calibration_and_detection():
    val_res = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2])
    baseline = FixedThresholdBaseline()
    thresh = baseline.fit_from_residuals(val_res, multiplier=3.0)
    assert thresh > 0

    res = baseline.detect(actual=[1.0, 5.0], predicted=[1.0, 1.0])
    assert not bool(res.loc[0, "is_anomaly"])
    assert bool(res.loc[1, "is_anomaly"])
