"""Sliding-window builder.

Transforms tidy per-cycle sensor frames into the fixed-length 3-D windows
(``[n_windows, window_size, n_features]``) consumed by the Temporal Fusion
Transformer and the Anomaly Transformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.cmapss_loader import feature_columns
from src.config import SENSOR


def build_windows(
    frame: pd.DataFrame,
    window_size: int | None = None,
    stride: int | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows grouped by ``unit_id``.

    Returns
    -------
    windows : ndarray ``[N, window_size, F]``
    rul     : ndarray ``[N]`` — RUL aligned to the last cycle of each window
    anomaly : ndarray ``[N, window_size]`` — per-step anomaly labels (0/1)
    """
    window_size = window_size or SENSOR.window_size
    stride = stride or SENSOR.window_stride
    feature_cols = feature_cols or feature_columns()

    windows, ruls, anomalies = [], [], []
    has_anomaly = "is_anomaly" in frame.columns

    for _, unit in frame.groupby("unit_id"):
        unit = unit.sort_values("cycle")
        values = unit[feature_cols].to_numpy(dtype=np.float32)
        rul = unit["rul"].to_numpy(dtype=np.float32) if "rul" in unit else None
        anom = (
            unit["is_anomaly"].to_numpy(dtype=np.float32)
            if has_anomaly
            else np.zeros(len(unit), dtype=np.float32)
        )
        n = len(unit)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end])
            if rul is not None:
                ruls.append(rul[end - 1])
            anomalies.append(anom[start:end])

    windows_arr = np.asarray(windows, dtype=np.float32)
    rul_arr = np.asarray(ruls, dtype=np.float32) if ruls else np.zeros(len(windows_arr))
    anom_arr = np.asarray(anomalies, dtype=np.float32)
    return windows_arr, rul_arr, anom_arr


def latest_window(
    frame: pd.DataFrame,
    window_size: int | None = None,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Return the most recent window for a single asset (for live scoring)."""
    window_size = window_size or SENSOR.window_size
    feature_cols = feature_cols or feature_columns()
    values = frame.sort_values("cycle")[feature_cols].to_numpy(dtype=np.float32)
    if len(values) < window_size:
        pad = np.repeat(values[:1], window_size - len(values), axis=0)
        values = np.vstack([pad, values])
    return values[-window_size:][None, ...]
