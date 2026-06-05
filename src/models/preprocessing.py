"""Shared scaling utilities for windowed sensor tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class WindowScaler:
    """Per-channel min-max scaler that operates on ``[N, T, F]`` tensors."""

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> WindowScaler:
        flat = windows.reshape(-1, windows.shape[-1])
        self.min_ = flat.min(axis=0)
        self.max_ = flat.max(axis=0)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        rng = (self.max_ - self.min_)
        rng[rng == 0] = 1.0
        return ((windows - self.min_) / rng).astype(np.float32)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"min": self.min_.tolist(), "max": self.max_.tolist()})
        )

    @classmethod
    def load(cls, path: str | Path) -> WindowScaler:
        data = json.loads(Path(path).read_text())
        scaler = cls()
        scaler.min_ = np.asarray(data["min"], dtype=np.float32)
        scaler.max_ = np.asarray(data["max"], dtype=np.float32)
        return scaler
