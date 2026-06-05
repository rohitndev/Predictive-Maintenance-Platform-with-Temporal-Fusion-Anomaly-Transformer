"""Custom domain sensor features for rotating machinery.

Adds engineering-informed degradation indicators on top of the automated
statistical features — cumulative damage, rolling health score, and
sensor-trend monotonicity — that are predictive of mechanical wear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_health_score(window: np.ndarray) -> float:
    """Normalised health score in [0, 1]; 1 = healthy, 0 = degraded.

    Built from the dispersion of the latest steps versus the window baseline.
    """
    baseline = window[: len(window) // 2]
    recent = window[len(window) // 2 :]
    drift = np.abs(recent.mean(axis=0) - baseline.mean(axis=0))
    score = 1.0 / (1.0 + drift.mean())
    return float(np.clip(score, 0.0, 1.0))


def monotonicity(series: np.ndarray) -> float:
    """Degradation monotonicity metric in [0, 1] (prognosability heuristic)."""
    diffs = np.diff(series)
    if len(diffs) == 0:
        return 0.0
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    return float(abs(pos - neg) / len(diffs))


def cumulative_damage(window: np.ndarray) -> float:
    """Integrated absolute change across all channels (proxy for fatigue)."""
    return float(np.sum(np.abs(np.diff(window, axis=0))))


def add_domain_features(features: pd.DataFrame, windows: np.ndarray) -> pd.DataFrame:
    """Append domain features to a per-window feature matrix."""
    features = features.copy().reset_index(drop=True)
    health = [rolling_health_score(w) for w in windows]
    damage = [cumulative_damage(w) for w in windows]
    mono = [monotonicity(w.mean(axis=1)) for w in windows]
    features["domain__health_score"] = health
    features["domain__cumulative_damage"] = damage
    features["domain__monotonicity"] = mono
    return features
