"""Tests for feature engineering and domain sensor features."""

from __future__ import annotations

import numpy as np

from data.cmapss_loader import feature_columns
from src.config import SENSOR, TFT
from src.features.feature_engineering import FeatureEngineer, extract_window_features
from src.features.sensor_features import (
    add_domain_features,
    cumulative_damage,
    monotonicity,
    rolling_health_score,
)


def _windows(n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, SENSOR.window_size, TFT.input_size)).astype(np.float32)


def test_extract_window_features_keys():
    feats = extract_window_features(_windows(1)[0], feature_columns())
    assert any(k.endswith("__mean") for k in feats)
    assert any("fft_coeff" in k for k in feats)


def test_feature_engineer_matrix_shape():
    fe = FeatureEngineer()
    matrix = fe.transform(_windows(8))
    assert len(matrix) == 8
    assert matrix.shape[1] >= len(feature_columns())


def test_domain_features_added():
    windows = _windows(6)
    fe = FeatureEngineer()
    matrix = add_domain_features(fe.transform(windows), windows)
    assert "domain__health_score" in matrix.columns
    assert "domain__cumulative_damage" in matrix.columns
    assert "domain__monotonicity" in matrix.columns


def test_health_score_bounds():
    w = _windows(1)[0]
    assert 0.0 <= rolling_health_score(w) <= 1.0
    assert cumulative_damage(w) >= 0.0
    assert 0.0 <= monotonicity(w.mean(axis=1)) <= 1.0
