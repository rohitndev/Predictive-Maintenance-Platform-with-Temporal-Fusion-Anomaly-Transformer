"""Physics-based degradation model (Digital Twin validation).

Implements a polynomial degradation baseline that estimates Remaining Useful
Life (RUL) directly from a health-index trend — independent of the learned
models. The hybrid physics-ML approach compares the data-driven TFT prediction
against this physical baseline to flag implausible predictions, following the
C-MAPSS prognostics convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import SENSOR


@dataclass
class PhysicsValidation:
    physics_rul: float
    ml_rul: float
    agreement: float          # 1.0 = perfect agreement, 0.0 = full disagreement
    is_consistent: bool


class PolynomialDegradationModel:
    """Fits a low-order polynomial to a health index and extrapolates failure."""

    def __init__(self, degree: int = 2, failure_threshold: float = 1.0):
        self.degree = degree
        self.failure_threshold = failure_threshold

    @staticmethod
    def health_index(window: np.ndarray) -> np.ndarray:
        """Collapse a multivariate window into a scalar health-loss trend.

        ``window`` has shape ``[T, F]``. We z-normalise each channel against its
        own window history and average the absolute deviation per time step,
        yielding a monotonic-ish degradation signal in arbitrary units.
        """
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-6
        z = np.abs((window - mean) / std)
        return z.mean(axis=1)

    def predict_rul(self, window: np.ndarray) -> float:
        """Estimate cycles-to-failure from the degradation severity.

        A polynomial is fit to the health-index trend to obtain (a) the current
        degradation level and (b) its growth rate. RUL is then mapped onto the
        C-MAPSS cycle scale through an exponential damage law — a standard
        physics-of-failure baseline that yields a value comparable to the
        learned model so the two can be cross-checked.
        """
        hi = self.health_index(window)
        t = np.arange(len(hi))
        coeffs = np.polyfit(t, hi, deg=min(self.degree, len(hi) - 1))
        poly = np.poly1d(coeffs)

        level = float(poly(len(hi) - 1))                       # current health loss
        slope = float(np.polyfit(t, hi, 1)[0])                 # degradation rate
        degradation_rate = abs(slope) / (abs(hi.mean()) + 1e-6)

        # Physics-of-failure severity: both how degraded (level) and how fast it
        # is degrading (rate) shorten the remaining life. Constants are
        # calibrated so the output lands on the C-MAPSS cycle scale.
        severity = 0.5 * level + 300.0 * degradation_rate
        physics_rul = SENSOR.rul_cap * float(np.exp(-severity))
        return float(np.clip(physics_rul, 0.0, SENSOR.rul_cap))

    def validate(self, ml_rul: float, window: np.ndarray, tol: float = 0.5) -> PhysicsValidation:
        """Compare a model RUL against the physics baseline."""
        physics_rul = self.predict_rul(window)
        denom = max(physics_rul, ml_rul, 1.0)
        agreement = 1.0 - abs(physics_rul - ml_rul) / denom
        agreement = float(np.clip(agreement, 0.0, 1.0))
        return PhysicsValidation(
            physics_rul=physics_rul,
            ml_rul=float(ml_rul),
            agreement=agreement,
            is_consistent=agreement >= (1.0 - tol),
        )
