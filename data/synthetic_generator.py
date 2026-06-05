"""Synthetic multivariate sensor generator (NASA C-MAPSS style).

Produces turbofan-engine-like run-to-failure trajectories with realistic
degradation, operating-condition variation, and injectable point / contextual
anomalies. Used as a self-contained, license-free stand-in for the NASA
C-MAPSS dataset so the platform runs end-to-end without external downloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SENSOR

# 21 sensors mirroring the C-MAPSS FD001 channel layout. Not every channel
# degrades — some are constant/noisy, exactly like the real dataset.
_DEGRADING_SENSORS = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16, 17, 19, 20]
_FLAT_SENSORS = [0, 5, 9, 10, 14, 18]


@dataclass
class GeneratorConfig:
    n_units: int = 80
    min_life: int = 130
    max_life: int = 320
    noise: float = 0.02
    seed: int = 42


def _degradation_curve(length: int, rng: np.random.Generator) -> np.ndarray:
    """Monotonic, slightly accelerating health-loss curve in [0, 1]."""
    t = np.linspace(0.0, 1.0, length)
    # exponent > 1 makes the failure accelerate near end-of-life
    exponent = rng.uniform(1.6, 2.6)
    curve = t**exponent
    return curve


def generate_unit(unit_id: int, rng: np.random.Generator, cfg: GeneratorConfig) -> pd.DataFrame:
    """Generate one run-to-failure trajectory for a single asset."""
    life = int(rng.integers(cfg.min_life, cfg.max_life))
    health_loss = _degradation_curve(life, rng)

    n_total = SENSOR.n_sensors
    data = np.zeros((life, n_total), dtype=np.float32)

    # Per-sensor degradation direction & magnitude.
    for s in range(n_total):
        baseline = rng.uniform(0.2, 0.8)
        if s in _DEGRADING_SENSORS:
            direction = rng.choice([-1.0, 1.0])
            magnitude = rng.uniform(0.25, 0.6)
            signal = baseline + direction * magnitude * health_loss
        else:
            signal = np.full(life, baseline, dtype=np.float32)
        noise = rng.normal(0.0, cfg.noise, size=life)
        data[:, s] = signal + noise

    # Operating-condition settings (3 columns) — bounded random walk.
    op_settings = np.zeros((life, SENSOR.n_op_settings), dtype=np.float32)
    for o in range(SENSOR.n_op_settings):
        walk = np.cumsum(rng.normal(0.0, 0.01, size=life))
        op_settings[:, o] = 0.5 + 0.1 * np.sin(np.linspace(0, 6, life)) + walk * 0.1

    cycles = np.arange(1, life + 1)
    rul = (life - cycles).astype(np.float32)
    rul = np.clip(rul, 0, SENSOR.rul_cap)

    cols = (
        [f"op_setting_{i + 1}" for i in range(SENSOR.n_op_settings)]
        + [f"sensor_{i + 1}" for i in range(n_total)]
    )
    frame = pd.DataFrame(
        np.hstack([op_settings, data]),
        columns=cols,
    )
    frame.insert(0, "unit_id", unit_id)
    frame.insert(1, "cycle", cycles)
    frame["rul"] = rul
    return frame


def inject_anomalies(
    frame: pd.DataFrame, rng: np.random.Generator, fraction: float = 0.04
) -> pd.DataFrame:
    """Inject labelled point (spike) and contextual (drift) anomalies."""
    frame = frame.copy()
    labels = np.zeros(len(frame), dtype=np.int64)
    sensor_cols = [c for c in frame.columns if c.startswith("sensor_")]
    values = {c: frame[c].to_numpy(dtype=np.float32).copy() for c in sensor_cols}

    n_anom = max(1, int(len(frame) * fraction))
    positions = rng.choice(len(frame), size=n_anom, replace=False)

    for pos in positions:
        col = rng.choice(sensor_cols)
        if rng.random() < 0.5:
            # point anomaly: sharp spike
            delta = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.2)
        else:
            # contextual anomaly: subtle deviation within normal range
            delta = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.3)
        values[col][pos] += np.float32(delta)
        labels[pos] = 1

    for c in sensor_cols:
        frame[c] = values[c]
    frame["is_anomaly"] = labels
    return frame


def generate_dataset(
    cfg: GeneratorConfig | None = None, with_anomalies: bool = True
) -> pd.DataFrame:
    """Generate the full multi-asset dataset as a tidy DataFrame."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for unit in range(1, cfg.n_units + 1):
        unit_frame = generate_unit(unit, rng, cfg)
        if with_anomalies:
            unit_frame = inject_anomalies(unit_frame, rng)
        frames.append(unit_frame)
    dataset = pd.concat(frames, ignore_index=True)
    return dataset


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {df['unit_id'].nunique()} assets, {len(df)} sensor cycles.")
    print(df.head())
