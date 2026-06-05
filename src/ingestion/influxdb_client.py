"""InfluxDB sensor TSDB client (optional).

Writes and reads 1 Hz sensor streams to/from InfluxDB Cloud. When no InfluxDB
credentials are configured (``CLOUD.influx_enabled`` is ``False``) the client
operates in an in-memory fallback mode so ingestion code paths stay testable
without a live time-series database.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import CLOUD

logger = logging.getLogger(__name__)


class SensorTSDBClient:
    """Thin wrapper over influxdb-client with a local fallback."""

    def __init__(self) -> None:
        self._client = None
        self._enabled = CLOUD.influx_enabled
        self._buffer: list[dict] = []
        if self._enabled:
            try:  # pragma: no cover - requires live InfluxDB
                from influxdb_client import InfluxDBClient

                self._client = InfluxDBClient(
                    url=CLOUD.influx_url,
                    token=CLOUD.influx_token,
                    org=CLOUD.influx_org,
                )
                logger.info("Connected to InfluxDB at %s", CLOUD.influx_url)
            except Exception as exc:  # pragma: no cover
                logger.warning("InfluxDB unavailable (%s); using local buffer.", exc)
                self._enabled = False

    @property
    def connected(self) -> bool:
        return self._enabled and self._client is not None

    def write_window(self, asset_id: str, frame: pd.DataFrame) -> int:
        """Publish a sensor frame for an asset. Returns number of points."""
        records = frame.assign(asset_id=asset_id).to_dict("records")
        if self.connected:  # pragma: no cover - requires live InfluxDB
            from influxdb_client import Point
            from influxdb_client.client.write_api import SYNCHRONOUS

            write_api = self._client.write_api(write_options=SYNCHRONOUS)
            points = []
            for rec in records:
                point = Point("sensor").tag("asset_id", asset_id)
                for key, value in rec.items():
                    if key.startswith(("sensor_", "op_setting_")):
                        point = point.field(key, float(value))
                points.append(point)
            write_api.write(bucket=CLOUD.influx_bucket, record=points)
            return len(points)
        # Fallback: buffer locally.
        self._buffer.extend(records)
        return len(records)

    def read_asset(self, asset_id: str) -> pd.DataFrame:
        """Read buffered/queried sensor history for an asset."""
        if self.connected:  # pragma: no cover - requires live InfluxDB
            query = (
                f'from(bucket:"{CLOUD.influx_bucket}") '
                f"|> range(start: -30d) "
                f'|> filter(fn:(r) => r.asset_id == "{asset_id}")'
            )
            return self._client.query_api().query_data_frame(query)
        rows = [r for r in self._buffer if r.get("asset_id") == asset_id]
        return pd.DataFrame(rows)
