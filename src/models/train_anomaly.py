"""Anomaly Transformer trainer.

Trains the Anomaly Transformer with the minimax association-discrepancy
objective and a reconstruction loss, then calibrates the anomaly-score
threshold and persists artifacts under ``artifacts/models/``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ANOMALY, MODEL_DIR, SENSOR, AnomalyConfig
from src.models.anomaly_transformer import (
    AnomalyTransformer,
    anomaly_score,
    association_discrepancy,
)
from src.models.preprocessing import WindowScaler

logger = logging.getLogger(__name__)

ANOMALY_WEIGHTS = MODEL_DIR / "anomaly.pt"
ANOMALY_SCALER = MODEL_DIR / "anomaly_scaler.json"
ANOMALY_META = MODEL_DIR / "anomaly_meta.json"


def train_anomaly(
    windows: np.ndarray,
    config: AnomalyConfig | None = None,
) -> dict:
    """Train the Anomaly Transformer and persist artifacts."""
    config = config or ANOMALY
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    scaler = WindowScaler()
    windows_scaled = scaler.fit_transform(windows)

    ds = TensorDataset(torch.tensor(windows_scaled))
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    model = AnomalyTransformer(win_size=SENSOR.window_size, config=config)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = torch.nn.MSELoss()

    history = []
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            optim.zero_grad()
            recon, series, prior = model(xb)
            recon_loss = mse(recon, xb)
            # Minimax association discrepancy: maximise discrepancy w.r.t the
            # series association, minimise w.r.t the prior (sign convention via k).
            assoc = association_discrepancy(series, prior).mean()
            loss = recon_loss - config.k * 0.01 * assoc
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ds)
        history.append({"epoch": epoch, "loss": epoch_loss})
        logger.info("[Anomaly] epoch %d loss=%.5f", epoch, epoch_loss)

    # Calibrate threshold from the score distribution on the training set.
    model.eval()
    scores = []
    with torch.no_grad():
        for (xb,) in loader:
            recon, series, prior = model(xb)
            s = anomaly_score(xb, recon, series, prior).mean(dim=1)
            scores.append(s.numpy())
    scores = np.concatenate(scores)
    threshold = float(np.quantile(scores, 1.0 - config.anomaly_ratio / 100.0))
    score_min, score_max = float(scores.min()), float(scores.max())

    torch.save(model.state_dict(), ANOMALY_WEIGHTS)
    scaler.save(ANOMALY_SCALER)
    ANOMALY_META.write_text(
        json.dumps({"threshold": threshold, "score_min": score_min, "score_max": score_max})
    )

    return {
        "weights": str(ANOMALY_WEIGHTS),
        "threshold": threshold,
        "history": history,
    }


def load_anomaly(
    config: AnomalyConfig | None = None,
) -> tuple[AnomalyTransformer, WindowScaler, dict]:
    """Load a trained Anomaly Transformer, scaler, and calibration metadata."""
    config = config or ANOMALY
    model = AnomalyTransformer(win_size=SENSOR.window_size, config=config)
    model.load_state_dict(torch.load(ANOMALY_WEIGHTS, map_location="cpu"))
    model.eval()
    scaler = WindowScaler.load(ANOMALY_SCALER)
    meta = json.loads(ANOMALY_META.read_text())
    return model, scaler, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.cmapss_loader import load_cmapss
    from src.ingestion.window_builder import build_windows

    df = load_cmapss()
    w, _, _ = build_windows(df)
    summary = train_anomaly(w)
    print(f"Anomaly Transformer trained. Threshold: {summary['threshold']:.4f}")
