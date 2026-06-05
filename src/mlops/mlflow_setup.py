"""MLflow model-registry helper (optional).

Registers trained models and promotes a champion. Falls back to a local JSON
registry under ``artifacts/`` when MLflow is not installed or no tracking URI
is configured, so model-versioning code paths always work.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from src.config import ARTIFACT_DIR, CLOUD

logger = logging.getLogger(__name__)

_LOCAL_REGISTRY = ARTIFACT_DIR / "model_registry.json"


class ModelRegistry:
    """Thin MLflow wrapper with a local-file fallback."""

    def __init__(self) -> None:
        self.uri = CLOUD.mlflow_tracking_uri
        self.enabled = bool(self.uri)
        if self.enabled:
            try:  # pragma: no cover - requires MLflow server
                import mlflow

                mlflow.set_tracking_uri(self.uri)
                self._mlflow = mlflow
                logger.info("MLflow tracking at %s", self.uri)
            except Exception as exc:  # pragma: no cover
                logger.warning("MLflow unavailable (%s); using local registry.", exc)
                self.enabled = False

    def register(self, name: str, version: str, metrics: dict, path: str) -> dict:
        """Register a model version with its evaluation metrics."""
        entry = {
            "name": name,
            "version": version,
            "metrics": metrics,
            "path": path,
            "registered_at": datetime.now(UTC).isoformat(),
            "stage": "champion",
        }
        if self.enabled:  # pragma: no cover - requires MLflow server
            with self._mlflow.start_run(run_name=f"{name}-{version}"):
                self._mlflow.log_metrics(metrics)
                self._mlflow.log_param("model_path", path)
        self._append_local(entry)
        return entry

    def _append_local(self, entry: dict) -> None:
        registry = []
        if _LOCAL_REGISTRY.exists():
            registry = json.loads(_LOCAL_REGISTRY.read_text())
        # Demote previous champions of the same model.
        for item in registry:
            if item["name"] == entry["name"]:
                item["stage"] = "archived"
        registry.append(entry)
        _LOCAL_REGISTRY.write_text(json.dumps(registry, indent=2))

    def champion(self, name: str) -> dict | None:
        if not _LOCAL_REGISTRY.exists():
            return None
        registry = json.loads(_LOCAL_REGISTRY.read_text())
        champs = [r for r in registry if r["name"] == name and r["stage"] == "champion"]
        return champs[-1] if champs else None


if __name__ == "__main__":
    reg = ModelRegistry()
    reg.register("tft-rul", "v1", {"val_rmse": 18.4}, "artifacts/models/tft.pt")
    print("Champion:", reg.champion("tft-rul"))
