"""Dual-model inference ensemble.

Combines the Temporal Fusion Transformer (probabilistic RUL), the Anomaly
Transformer (point + contextual anomaly score), and the physics-based digital
twin validation into a single asset-health assessment. This is the object the
FastAPI serving layer and the CMMS trigger consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from data.cmapss_loader import feature_columns
from src.config import SENSOR, TRIGGER
from src.features.physics_model import PolynomialDegradationModel
from src.models.anomaly_transformer import anomaly_score
from src.models.preprocessing import WindowScaler


@dataclass
class AssetAssessment:
    asset_id: str
    rul_p10: float
    rul_p50: float
    rul_p90: float
    anomaly_score: float
    is_anomalous: bool
    physics_rul: float
    physics_agreement: float
    physics_consistent: bool
    health_score: float
    top_drivers: list[dict]
    needs_maintenance: bool

    def to_dict(self) -> dict:
        return asdict(self)


class PredictiveMaintenanceEnsemble:
    """Holds the trained models and produces asset assessments."""

    def __init__(
        self,
        tft_model,
        tft_scaler: WindowScaler,
        anomaly_model,
        anomaly_scaler: WindowScaler,
        anomaly_meta: dict,
    ):
        self.tft_model = tft_model
        self.tft_scaler = tft_scaler
        self.anomaly_model = anomaly_model
        self.anomaly_scaler = anomaly_scaler
        self.anomaly_meta = anomaly_meta
        self.physics = PolynomialDegradationModel()
        self.channels = feature_columns()

    def _predict_rul(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.tft_scaler.transform(window[None, ...])
        with torch.no_grad():
            quantiles, weights = self.tft_model(torch.tensor(scaled))
        rul = quantiles[0].numpy() * SENSOR.rul_cap
        return np.clip(rul, 0, SENSOR.rul_cap), weights[0].numpy()

    def _predict_anomaly(self, window: np.ndarray) -> float:
        scaled = self.anomaly_scaler.transform(window[None, ...])
        x = torch.tensor(scaled)
        with torch.no_grad():
            recon, series, prior = self.anomaly_model(x)
            score = anomaly_score(x, recon, series, prior).mean().item()
        lo = self.anomaly_meta["score_min"]
        hi = self.anomaly_meta["score_max"]
        norm = (score - lo) / (hi - lo + 1e-8)
        return float(np.clip(norm, 0.0, 1.0))

    def assess(self, asset_id: str, window: np.ndarray) -> AssetAssessment:
        """Run the full dual-model + physics assessment on one window."""
        rul, weights = self._predict_rul(window)
        p10, p50, p90 = sorted(rul.tolist())          # enforce monotone quantiles
        anomaly = self._predict_anomaly(window)
        physics = self.physics.validate(p50, window)

        order = np.argsort(weights)[::-1][:5]
        top_drivers = [
            {"sensor": self.channels[i], "importance": round(float(weights[i]), 4)}
            for i in order
        ]
        health = round(float(np.clip(p50 / SENSOR.rul_cap, 0.0, 1.0)), 4)
        is_anom = anomaly >= TRIGGER.anomaly_threshold
        needs_maint = (p50 < TRIGGER.rul_alert_hours) and is_anom

        return AssetAssessment(
            asset_id=asset_id,
            rul_p10=round(p10, 2),
            rul_p50=round(p50, 2),
            rul_p90=round(p90, 2),
            anomaly_score=round(anomaly, 4),
            is_anomalous=is_anom,
            physics_rul=round(physics.physics_rul, 2),
            physics_agreement=round(physics.agreement, 4),
            physics_consistent=physics.is_consistent,
            health_score=health,
            top_drivers=top_drivers,
            needs_maintenance=needs_maint,
        )


def load_ensemble() -> PredictiveMaintenanceEnsemble:
    """Convenience loader that wires trained TFT + Anomaly models together."""
    from src.models.train_anomaly import load_anomaly
    from src.models.train_tft import load_tft

    tft_model, tft_scaler = load_tft()
    anomaly_model, anomaly_scaler, anomaly_meta = load_anomaly()
    return PredictiveMaintenanceEnsemble(
        tft_model, tft_scaler, anomaly_model, anomaly_scaler, anomaly_meta
    )
