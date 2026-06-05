"""Tests for the CMMS agent, work-order generator, and SAP PM mock."""

from __future__ import annotations

import pytest

from src.agent.cmms_agent import CMMSAgent
from src.agent.sap_pm_api import SAPMaintenanceClient
from src.agent.work_order import build_work_order


@pytest.fixture
def assessment():
    return {
        "asset_id": "ENGINE-042",
        "rul_p10": 6.0,
        "rul_p50": 18.0,
        "rul_p90": 35.0,
        "anomaly_score": 0.82,
        "is_anomalous": True,
        "physics_rul": 20.0,
        "physics_consistent": True,
        "health_score": 0.14,
        "needs_maintenance": True,
        "top_drivers": [
            {"sensor": "sensor_3", "importance": 0.22},
            {"sensor": "sensor_12", "importance": 0.17},
            {"sensor": "sensor_20", "importance": 0.10},
        ],
    }


def test_build_work_order_priority(assessment):
    wo = build_work_order(assessment)
    assert wo.priority == "P2-HIGH"          # 18h -> P2
    assert wo.asset_id == "ENGINE-042"
    assert wo.required_parts
    assert wo.technician_skills
    assert wo.work_order_id.startswith("WO-")


def test_sap_pm_creates_and_lists():
    client = SAPMaintenanceClient()
    confirmation = client.create_work_order({"work_order_id": "WO-TEST", "priority": "P1"})
    assert confirmation["sap_pm_status"] == "CREATED"
    assert confirmation["sap_notification_id"]
    assert client.get_work_order("WO-TEST") is not None
    assert len(client.list_work_orders()) == 1


def test_agent_triggers_when_needed(assessment):
    agent = CMMSAgent()
    confirmation = agent.process_assessment(assessment)
    assert confirmation is not None
    assert confirmation["sap_pm_status"] == "CREATED"


def test_agent_skips_when_healthy(assessment):
    healthy = {**assessment, "needs_maintenance": False, "is_anomalous": False}
    agent = CMMSAgent()
    assert agent.process_assessment(healthy) is None
