"""End-to-end training pipeline.

Loads (or generates) C-MAPSS-style sensor data, builds sliding windows, trains
the Temporal Fusion Transformer and the Anomaly Transformer, and registers the
resulting models in the (local/MLflow) model registry.

Usage:
    python -m scripts.train_pipeline            # full training
    python -m scripts.train_pipeline --quick    # fast smoke training
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from data.cmapss_loader import load_cmapss
from src.config import ANOMALY, TFT
from src.ingestion.window_builder import build_windows
from src.mlops.mlflow_setup import ModelRegistry
from src.models.train_anomaly import train_anomaly
from src.models.train_tft import train_tft

logger = logging.getLogger(__name__)


def run_training(quick: bool = False, use_wandb: bool = False) -> dict:
    """Run the full training pipeline and return a summary."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Loading sensor data...")
    df = load_cmapss()
    windows, rul, _ = build_windows(df)
    logger.info("Built %d windows of shape %s", len(windows), windows.shape[1:])

    if quick:
        idx = np.random.RandomState(0).permutation(len(windows))[:1500]
        windows, rul = windows[idx], rul[idx]
        TFT.max_epochs = 2
        ANOMALY.max_epochs = 2

    logger.info("Training Temporal Fusion Transformer (RUL)...")
    tft_summary = train_tft(windows, rul, use_wandb=use_wandb)

    logger.info("Training Anomaly Transformer...")
    anomaly_summary = train_anomaly(windows)

    registry = ModelRegistry()
    registry.register(
        "tft-rul", "v1", {"val_rmse": tft_summary["final_val_rmse"]}, tft_summary["weights"]
    )
    registry.register(
        "anomaly-transformer", "v1", {"threshold": anomaly_summary["threshold"]},
        anomaly_summary["weights"],
    )

    logger.info(
        "Training complete. TFT val RMSE=%.2f cycles | anomaly threshold=%.4f",
        tft_summary["final_val_rmse"],
        anomaly_summary["threshold"],
    )
    return {"tft": tft_summary, "anomaly": anomaly_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive-maintenance models")
    parser.add_argument("--quick", action="store_true", help="Fast smoke training")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()
    run_training(quick=args.quick, use_wandb=args.wandb)
