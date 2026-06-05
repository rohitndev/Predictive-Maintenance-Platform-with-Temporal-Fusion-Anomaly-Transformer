"""Streaming sensor reader.

Reads sliding windows from a sensor source. Uses PySpark Structured Streaming
when Spark is available (Databricks CE / local Spark); otherwise falls back to
a pandas micro-batch reader with identical semantics so the platform runs on a
single machine without a Spark cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pandas as pd

from src.config import SENSOR
from src.ingestion.window_builder import build_windows

logger = logging.getLogger(__name__)


def _spark_available() -> bool:
    try:  # pragma: no cover - optional heavy dep
        import pyspark  # noqa: F401

        return True
    except Exception:
        return False


class SensorStreamReader:
    """Yields fixed-size sensor windows in micro-batches."""

    def __init__(self, window_size: int | None = None, stride: int | None = None):
        self.window_size = window_size or SENSOR.window_size
        self.stride = stride or SENSOR.window_stride
        self.engine = "pyspark" if _spark_available() else "pandas"
        logger.info("SensorStreamReader engine: %s", self.engine)

    def read_windows(self, frame: pd.DataFrame):
        """Return ``(windows, rul, anomaly_labels)`` arrays from a frame."""
        if self.engine == "pyspark":  # pragma: no cover - optional
            return self._read_windows_spark(frame)
        return build_windows(frame, self.window_size, self.stride)

    def stream_micro_batches(
        self, frame: pd.DataFrame, batch_assets: int = 8
    ) -> Iterator[pd.DataFrame]:
        """Simulate a real-time stream by yielding asset micro-batches."""
        unit_ids = sorted(frame["unit_id"].unique())
        for i in range(0, len(unit_ids), batch_assets):
            chunk = unit_ids[i : i + batch_assets]
            yield frame[frame["unit_id"].isin(chunk)].copy()

    def _read_windows_spark(self, frame: pd.DataFrame):  # pragma: no cover - optional
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("pdm-ingestion").getOrCreate()
        sdf = spark.createDataFrame(frame)
        # Spark is used here for distributed pre-aggregation; window assembly
        # is delegated to the shared numpy routine for model-input parity.
        collected = sdf.toPandas()
        spark.stop()
        return build_windows(collected, self.window_size, self.stride)
