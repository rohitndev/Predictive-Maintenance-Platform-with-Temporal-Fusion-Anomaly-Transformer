"""NASA C-MAPSS Turbofan Engine Degradation dataset loader.

If the official C-MAPSS text files (``train_FD001.txt`` etc.) are present in
``data/raw/``, they are parsed into the canonical schema. Otherwise the loader
transparently falls back to the synthetic generator so downstream code always
receives a consistent DataFrame.

Dataset reference: NASA C-MAPSS Turbofan Engine Degradation Simulation
(https://www.nasa.gov/intelligent-systems-division — Prognostics Data Repository).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic_generator import GeneratorConfig, generate_dataset
from src.config import DATA_DIR, SENSOR

_RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i + 1}" for i in range(SENSOR.n_op_settings)]
    + [f"sensor_{i + 1}" for i in range(SENSOR.n_sensors)]
)


def _compute_rul(frame: pd.DataFrame) -> pd.DataFrame:
    max_cycle = frame.groupby("unit_id")["cycle"].transform("max")
    frame["rul"] = (max_cycle - frame["cycle"]).clip(upper=SENSOR.rul_cap).astype(np.float32)
    return frame


def load_cmapss(subset: str = "FD001") -> pd.DataFrame:
    """Load a C-MAPSS subset, falling back to synthetic data when absent."""
    raw_path = Path(DATA_DIR) / f"train_{subset}.txt"
    if raw_path.exists():
        frame = pd.read_csv(raw_path, sep=r"\s+", header=None, engine="python")
        frame = frame.dropna(axis=1, how="all")
        frame.columns = _RAW_COLUMNS[: frame.shape[1]]
        frame = _compute_rul(frame)
        frame["is_anomaly"] = 0
        return frame

    # Fallback: synthetic C-MAPSS-style data.
    return generate_dataset(GeneratorConfig(), with_anomalies=True)


def feature_columns() -> list[str]:
    """Return the model input columns (operating settings + sensors)."""
    return [f"op_setting_{i + 1}" for i in range(SENSOR.n_op_settings)] + [
        f"sensor_{i + 1}" for i in range(SENSOR.n_sensors)
    ]


if __name__ == "__main__":
    df = load_cmapss()
    print(f"Loaded {df['unit_id'].nunique()} units / {len(df)} rows.")
    print(df[feature_columns()].describe().T.head())
