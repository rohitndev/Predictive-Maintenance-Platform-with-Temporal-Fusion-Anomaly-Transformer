"""Shared pytest fixtures.

Trains a small model set once per test session so the model/API/agent tests run
against real trained artifacts without retraining for every test.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.cmapss_loader import load_cmapss
from src.ingestion.window_builder import build_windows


@pytest.fixture(scope="session")
def dataset():
    return load_cmapss()


@pytest.fixture(scope="session")
def windows(dataset):
    w, r, a = build_windows(dataset)
    return w, r, a


@pytest.fixture(scope="session")
def trained_models(windows):
    """Train tiny TFT + Anomaly models once for the whole session."""
    from src.config import ANOMALY, TFT
    from src.models.train_anomaly import train_anomaly
    from src.models.train_tft import train_tft

    w, r, _ = windows
    idx = np.random.RandomState(0).permutation(len(w))[:600]
    TFT.max_epochs = 1
    ANOMALY.max_epochs = 1
    train_tft(w[idx], r[idx])
    train_anomaly(w[idx])
    return True


@pytest.fixture(scope="session")
def ensemble(trained_models):
    from src.models.ensemble import load_ensemble

    return load_ensemble()
