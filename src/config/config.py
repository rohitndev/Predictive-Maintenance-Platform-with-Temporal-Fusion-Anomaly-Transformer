"""Central configuration for the Predictive Maintenance Platform.

All tunable parameters (paths, model hyper-parameters, thresholds, and optional
cloud credentials) are resolved here. Cloud settings are read from environment
variables so the platform runs fully offline when no credentials are supplied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
OUTPUT_DIR = ARTIFACT_DIR / "output"

for _d in (DATA_DIR, ARTIFACT_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sensor / window configuration (NASA C-MAPSS style)
# --------------------------------------------------------------------------- #
@dataclass
class SensorConfig:
    """Defines the shape of the multivariate sensor stream."""

    n_sensors: int = 21
    n_op_settings: int = 3
    sampling_hz: float = 1.0
    window_size: int = 50           # sliding window length (cycles)
    window_stride: int = 1
    rul_cap: int = 125              # piece-wise linear RUL clip (C-MAPSS convention)


# --------------------------------------------------------------------------- #
# Temporal Fusion Transformer hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class TFTConfig:
    input_size: int = 24            # n_sensors + op_settings
    hidden_size: int = 64
    attention_heads: int = 4
    dropout: float = 0.1
    lstm_layers: int = 1
    quantiles: tuple = (0.1, 0.5, 0.9)   # P10 / P50 / P90
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 6
    seed: int = 42


# --------------------------------------------------------------------------- #
# Anomaly Transformer hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyConfig:
    input_size: int = 24
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 5
    anomaly_ratio: float = 1.0      # percentage of points expected anomalous
    k: float = 3.0                  # association-discrepancy weighting
    seed: int = 42


# --------------------------------------------------------------------------- #
# Decision thresholds that drive the CMMS work-order trigger
# --------------------------------------------------------------------------- #
@dataclass
class TriggerConfig:
    rul_alert_hours: int = 48       # trigger when P50 RUL < 48h
    anomaly_threshold: float = 0.55  # normalised anomaly score in [0, 1]


# --------------------------------------------------------------------------- #
# Optional cloud / external service configuration (graceful fallback)
# --------------------------------------------------------------------------- #
@dataclass
class CloudConfig:
    # InfluxDB (sensor TSDB)
    influx_url: str = field(default_factory=lambda: os.getenv("INFLUXDB_URL", ""))
    influx_token: str = field(default_factory=lambda: os.getenv("INFLUXDB_TOKEN", ""))
    influx_org: str = field(default_factory=lambda: os.getenv("INFLUXDB_ORG", ""))
    influx_bucket: str = field(
        default_factory=lambda: os.getenv("INFLUXDB_BUCKET", "sensors")
    )

    # AWS (S3 artifacts / Lambda serving)
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    aws_s3_bucket: str = field(default_factory=lambda: os.getenv("AWS_S3_BUCKET", ""))

    # Groq (CMMS LLM agent)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    )

    # Experiment tracking
    wandb_api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "")
    )

    @property
    def influx_enabled(self) -> bool:
        return bool(self.influx_url and self.influx_token)

    @property
    def aws_enabled(self) -> bool:
        return bool(self.aws_s3_bucket)

    @property
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)


SENSOR = SensorConfig()
TFT = TFTConfig()
ANOMALY = AnomalyConfig()
TRIGGER = TriggerConfig()
CLOUD = CloudConfig()
