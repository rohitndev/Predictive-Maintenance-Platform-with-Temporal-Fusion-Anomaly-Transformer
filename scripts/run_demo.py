"""End-to-end demonstration of the predictive-maintenance flow.

Streams synthetic sensor windows through the dual-model ensemble, validates
each prediction against the physics digital twin, and triggers the CMMS agent
for any asset within the maintenance horizon — printing a full asset-health
report to the terminal. Trains models first if no artifacts are present.

Usage:
    python -m scripts.run_demo
"""

from __future__ import annotations

import logging

from data.cmapss_loader import load_cmapss
from src.agent.cmms_agent import CMMSAgent
from src.ingestion.stream_reader import SensorStreamReader
from src.ingestion.window_builder import latest_window
from src.models.ensemble import load_ensemble

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> None:
    _banner("PREDICTIVE MAINTENANCE PLATFORM — LIVE ASSET HEALTH SCAN")

    try:
        ensemble = load_ensemble()
    except Exception:
        logger.info("No trained models found — running quick training first...\n")
        from scripts.train_pipeline import run_training

        run_training(quick=True)
        ensemble = load_ensemble()

    agent = CMMSAgent()
    df = load_cmapss()
    reader = SensorStreamReader()

    triggered = 0
    units = sorted(df["unit_id"].unique())[:12]
    print(f"\nIngestion engine: {reader.engine}")
    print(f"Scanning {len(units)} assets through the dual-model ensemble...\n")
    print(f"{'ASSET':<12}{'RUL P50':>9}{'P10-P90':>14}{'ANOMALY':>9}{'PHYS':>7}{'STATUS':>14}")
    print("-" * 70)

    for unit_id in units:
        unit = df[df["unit_id"] == unit_id]
        window = latest_window(unit)[0]
        a = ensemble.assess(f"ENGINE-{unit_id:03d}", window)
        status = "MAINTENANCE" if a.needs_maintenance else ("WATCH" if a.is_anomalous else "OK")
        ci = f"{a.rul_p10:.0f}-{a.rul_p90:.0f}h"
        phys = "ok" if a.physics_consistent else "diverge"
        print(
            f"{a.asset_id:<12}{a.rul_p50:>7.0f}h{ci:>14}{a.anomaly_score:>9.3f}"
            f"{phys:>7}{status:>14}"
        )

        confirmation = agent.process_assessment(a.to_dict())
        if confirmation:
            triggered += 1
            _banner(f"CMMS WORK ORDER FILED — {a.asset_id}")
            print(f"  Work Order : {confirmation['work_order_id']}")
            print(f"  Priority   : {confirmation['priority']}")
            print(f"  SAP Notif. : {confirmation['sap_notification_id']}")
            print(f"  Scheduled  : {confirmation['scheduled_start']}")
            print(f"  Summary    : {confirmation['summary']}")
            print(f"  Action     : {confirmation['recommended_action']}")

    _banner("SCAN COMPLETE")
    print(f"  Assets scanned        : {len(units)}")
    print(f"  Work orders triggered : {triggered}")
    print(f"  CMMS queue size       : {len(agent.cmms.list_work_orders())}\n")


if __name__ == "__main__":
    main()
