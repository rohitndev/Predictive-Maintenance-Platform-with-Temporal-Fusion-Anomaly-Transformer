"""SAP PM / ServiceMax CMMS API client (mock).

Simulates the Computerized Maintenance Management System integration: creating
maintenance notifications/work orders and scheduling technicians. When a real
CMMS endpoint is configured via ``CMMS_API_URL`` the same methods issue HTTP
calls; otherwise an in-memory store records the orders so the integration is
fully exercisable offline.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


class SAPMaintenanceClient:
    """Mock SAP Plant Maintenance (PM) work-order API."""

    def __init__(self) -> None:
        self.endpoint = os.getenv("CMMS_API_URL", "")
        self.enabled = bool(self.endpoint)
        self._orders: dict[str, dict] = {}

    def create_work_order(self, work_order: dict) -> dict:
        """Create a work order in the CMMS and return the confirmation."""
        if self.enabled:  # pragma: no cover - requires live CMMS
            import requests

            resp = requests.post(
                f"{self.endpoint}/work-orders", json=work_order, timeout=10
            )
            resp.raise_for_status()
            return resp.json()

        # Offline mock: assign a scheduled window and persist locally.
        now = datetime.now(UTC)
        window_start = now + timedelta(hours=2)
        confirmation = {
            **work_order,
            "sap_pm_status": "CREATED",
            "sap_notification_id": f"NOTIF-{len(self._orders) + 1:05d}",
            "scheduled_start": window_start.isoformat(),
            "scheduled_end": (window_start + timedelta(hours=4)).isoformat(),
            "created_at": now.isoformat(),
        }
        self._orders[work_order["work_order_id"]] = confirmation
        logger.info(
            "CMMS work order %s created (notification %s)",
            work_order["work_order_id"],
            confirmation["sap_notification_id"],
        )
        return confirmation

    def get_work_order(self, work_order_id: str) -> dict | None:
        return self._orders.get(work_order_id)

    def list_work_orders(self) -> list[dict]:
        return list(self._orders.values())
