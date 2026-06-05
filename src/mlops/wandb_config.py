"""Weights & Biases experiment tracking wrapper.

Provides a uniform tracking interface that becomes a no-op when W&B is not
installed or no API key is configured — so training scripts never fail because
of a missing experiment tracker.
"""

from __future__ import annotations

import logging

from src.config import CLOUD

logger = logging.getLogger(__name__)


class WandbTracker:
    """Lightweight, optional Weights & Biases tracker."""

    def __init__(self, enabled: bool = False, project: str = "predictive-maintenance"):
        self.project = project
        self.enabled = enabled and CLOUD.wandb_api_key != ""
        self._run = None

    def init(self, name: str, config: dict) -> None:
        if not self.enabled:
            return
        try:  # pragma: no cover - requires W&B account
            import wandb

            self._run = wandb.init(project=self.project, name=name, config=config)
            logger.info("W&B run started: %s", name)
        except Exception as exc:  # pragma: no cover
            logger.warning("W&B unavailable (%s); tracking disabled.", exc)
            self.enabled = False

    def log(self, metrics: dict) -> None:
        if self.enabled and self._run is not None:  # pragma: no cover
            import wandb

            wandb.log(metrics)

    def finish(self) -> None:
        if self.enabled and self._run is not None:  # pragma: no cover
            import wandb

            wandb.finish()
            self._run = None
