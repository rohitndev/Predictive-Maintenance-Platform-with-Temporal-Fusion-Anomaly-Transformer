"""Tests for the TFT, Anomaly Transformer, physics model, and ensemble."""

from __future__ import annotations

import numpy as np
import torch

from src.config import SENSOR, TFT
from src.features.physics_model import PolynomialDegradationModel
from src.models.anomaly_transformer import AnomalyTransformer, anomaly_score
from src.models.tft import QuantileLoss, TemporalFusionTransformer


def _dummy_window(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((SENSOR.window_size, TFT.input_size)).astype(np.float32)


def test_tft_forward_quantiles():
    model = TemporalFusionTransformer(TFT)
    x = torch.tensor(_dummy_window()[None, ...])
    preds, weights = model(x)
    assert preds.shape == (1, len(TFT.quantiles))
    assert weights.shape == (1, TFT.input_size)
    # variable-selection weights form a distribution
    assert abs(float(weights.sum()) - 1.0) < 1e-3


def test_quantile_loss_positive():
    loss = QuantileLoss(list(TFT.quantiles))
    preds = torch.rand(8, 3)
    target = torch.rand(8)
    assert float(loss(preds, target)) >= 0.0


def test_anomaly_transformer_forward():
    model = AnomalyTransformer(win_size=SENSOR.window_size)
    x = torch.tensor(_dummy_window()[None, ...])
    recon, series, prior = model(x)
    assert recon.shape == x.shape
    score = anomaly_score(x, recon, series, prior)
    assert score.shape == (1, SENSOR.window_size)
    assert float(score.min()) >= 0.0


def test_physics_validation_in_range():
    pm = PolynomialDegradationModel()
    window = _dummy_window(1)
    rul = pm.predict_rul(window)
    assert 0.0 <= rul <= SENSOR.rul_cap
    result = pm.validate(30.0, window)
    assert 0.0 <= result.agreement <= 1.0


def test_ensemble_assessment_fields(ensemble, dataset):
    from src.ingestion.window_builder import latest_window

    unit = dataset[dataset["unit_id"] == dataset["unit_id"].max()]
    window = latest_window(unit)[0]
    a = ensemble.assess("ENGINE-TEST", window)
    assert a.rul_p10 <= a.rul_p50 <= a.rul_p90
    assert 0.0 <= a.anomaly_score <= 1.0
    assert len(a.top_drivers) == 5
    assert isinstance(a.needs_maintenance, bool)
