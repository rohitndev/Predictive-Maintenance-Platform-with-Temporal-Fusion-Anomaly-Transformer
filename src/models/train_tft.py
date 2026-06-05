"""Temporal Fusion Transformer trainer.

Trains the TFT on windowed sensor data for probabilistic RUL prediction,
optionally logging metrics to Weights & Biases, and persists the model state
plus the fitted scaler under ``artifacts/models/``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODEL_DIR, SENSOR, TFT, TFTConfig
from src.mlops.wandb_config import WandbTracker
from src.models.preprocessing import WindowScaler
from src.models.tft import QuantileLoss, TemporalFusionTransformer

logger = logging.getLogger(__name__)

TFT_WEIGHTS = MODEL_DIR / "tft.pt"
TFT_SCALER = MODEL_DIR / "tft_scaler.json"


def train_tft(
    windows: np.ndarray,
    rul: np.ndarray,
    config: TFTConfig | None = None,
    use_wandb: bool = False,
) -> dict:
    """Train the TFT and persist artifacts. Returns a training summary."""
    config = config or TFT
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    scaler = WindowScaler()
    windows_scaled = scaler.fit_transform(windows)
    rul_norm = (rul / SENSOR.rul_cap).astype(np.float32)

    n = len(windows_scaled)
    split = int(n * 0.85)
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:split], perm[split:]

    def loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(windows_scaled[idx]), torch.tensor(rul_norm[idx])
        )
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(train_idx, True), loader(val_idx, False)

    model = TemporalFusionTransformer(config)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = QuantileLoss(list(config.quantiles))

    tracker = WandbTracker(enabled=use_wandb)
    tracker.init("tft-rul", config.__dict__)

    history = []
    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optim.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        val_loss, val_rmse = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds, _ = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                p50 = preds[:, 1] * SENSOR.rul_cap
                target = yb * SENSOR.rul_cap
                val_rmse += torch.sum((p50 - target) ** 2).item()
        val_loss /= len(val_idx)
        val_rmse = float(np.sqrt(val_rmse / len(val_idx)))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse": val_rmse})
        tracker.log({"train_loss": train_loss, "val_loss": val_loss, "val_rmse": val_rmse})
        logger.info(
            "[TFT] epoch %d train=%.4f val=%.4f rmse=%.2f", epoch, train_loss, val_loss, val_rmse
        )

    torch.save(model.state_dict(), TFT_WEIGHTS)
    scaler.save(TFT_SCALER)
    tracker.finish()

    return {
        "weights": str(TFT_WEIGHTS),
        "scaler": str(TFT_SCALER),
        "final_val_rmse": history[-1]["val_rmse"],
        "history": history,
    }


def load_tft(config: TFTConfig | None = None) -> tuple[TemporalFusionTransformer, WindowScaler]:
    """Load a trained TFT model and its scaler."""
    config = config or TFT
    model = TemporalFusionTransformer(config)
    model.load_state_dict(torch.load(TFT_WEIGHTS, map_location="cpu"))
    model.eval()
    scaler = WindowScaler.load(TFT_SCALER)
    return model, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.cmapss_loader import load_cmapss
    from src.ingestion.window_builder import build_windows

    df = load_cmapss()
    w, r, _ = build_windows(df)
    summary = train_tft(w, r)
    print(f"TFT trained. Final validation RMSE: {summary['final_val_rmse']:.2f} cycles")
