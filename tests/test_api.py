"""Integration tests for the FastAPI serving layer."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import SENSOR


@pytest.fixture(scope="module")
def client(trained_models):
    from api.main import app

    with TestClient(app) as c:
        yield c


def _payload(asset_id: str = "ENGINE-001") -> dict:
    rng = np.random.default_rng(0)
    readings = []
    for _ in range(SENSOR.window_size):
        readings.append(
            {
                "op_settings": rng.random(SENSOR.n_op_settings).tolist(),
                "sensors": rng.random(SENSOR.n_sensors).tolist(),
            }
        )
    return {"asset_id": asset_id, "readings": readings}


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert "cloud" in body


def test_predict_rul_endpoint(client):
    resp = client.post("/predict/rul", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["rul_p10"] <= body["rul_p50"] <= body["rul_p90"]
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert len(body["top_drivers"]) == 5


def test_predict_anomaly_endpoint(client):
    resp = client.post("/predict/anomaly", json=_payload())
    assert resp.status_code == 200
    assert "anomaly_score" in resp.json()


def test_maintenance_trigger_force(client):
    resp = client.post("/maintenance/trigger?force=true", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] is True
    assert body["work_order"]["work_order_id"].startswith("WO-")


def test_batch_endpoint(client):
    payload = {"assets": [_payload("ENGINE-001"), _payload("ENGINE-002")]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["assessments"]) == 2
    assert "work_orders_triggered" in body


def test_validation_error_on_bad_shape(client):
    bad = {"asset_id": "X", "readings": [{"op_settings": [0.1], "sensors": [0.2]}]}
    resp = client.post("/predict/rul", json=bad)
    assert resp.status_code == 422
