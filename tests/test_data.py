"""Tests for data generation, loading, and windowing."""

from __future__ import annotations

import numpy as np

from data.cmapss_loader import feature_columns
from data.synthetic_generator import GeneratorConfig, generate_dataset
from src.config import SENSOR
from src.ingestion.window_builder import build_windows, latest_window


def test_synthetic_dataset_shape():
    df = generate_dataset(GeneratorConfig(n_units=5))
    assert df["unit_id"].nunique() == 5
    assert "rul" in df.columns
    assert "is_anomaly" in df.columns
    assert (df["rul"] >= 0).all()
    assert df["rul"].max() <= SENSOR.rul_cap


def test_feature_columns_count():
    cols = feature_columns()
    assert len(cols) == SENSOR.n_op_settings + SENSOR.n_sensors


def test_window_shapes(dataset):
    w, r, a = build_windows(dataset)
    assert w.ndim == 3
    assert w.shape[1] == SENSOR.window_size
    assert w.shape[2] == SENSOR.n_op_settings + SENSOR.n_sensors
    assert len(r) == len(w)
    assert a.shape == (len(w), SENSOR.window_size)


def test_latest_window_pads_short_history(dataset):
    unit = dataset[dataset["unit_id"] == dataset["unit_id"].iloc[0]].head(5)
    win = latest_window(unit)
    assert win.shape == (1, SENSOR.window_size, len(feature_columns()))
    assert not np.isnan(win).any()
