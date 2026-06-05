"""Maintenance work-order data model and template generator.

Produces a structured CMMS work order from an asset assessment. The
template-based generator is the deterministic fallback used whenever the LLM
agent is unavailable, guaranteeing the platform always emits a valid work
order.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field


def _priority_from_rul(rul_hours: float) -> str:
    if rul_hours < 12:
        return "P1-CRITICAL"
    if rul_hours < 24:
        return "P2-HIGH"
    if rul_hours < 48:
        return "P3-MEDIUM"
    return "P4-LOW"


# Maps a degrading sensor family to likely failing components / required parts.
_COMPONENT_MAP = {
    "sensor_2": ("LPC outlet temperature", ["temp probe T24", "thermal paste"]),
    "sensor_3": ("HPC outlet temperature", ["temp probe T30", "seal kit"]),
    "sensor_4": ("LPT outlet temperature", ["temp probe T50"]),
    "sensor_7": ("HPC outlet pressure", ["pressure transducer Ps30"]),
    "sensor_11": ("HPC static pressure", ["pressure transducer"]),
    "sensor_12": ("fuel flow ratio", ["fuel metering valve", "filter element"]),
    "sensor_15": ("bypass ratio actuator", ["VBV actuator", "o-ring set"]),
    "sensor_17": ("bleed enthalpy", ["bleed valve assembly"]),
    "sensor_20": ("HPT coolant bleed", ["coolant line", "gasket"]),
    "sensor_21": ("LPT coolant bleed", ["coolant line"]),
}


@dataclass
class WorkOrder:
    work_order_id: str
    asset_id: str
    priority: str
    predicted_failure_hours: float
    confidence_interval: str
    suspected_components: list[str]
    required_parts: list[str]
    technician_skills: list[str]
    summary: str
    recommended_action: str
    generated_by: str = "template"
    sap_pm_status: str = "PENDING"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _wo_id(asset_id: str, rul_p50: float) -> str:
    digest = hashlib.sha1(f"{asset_id}-{rul_p50}".encode()).hexdigest()[:8].upper()
    return f"WO-{digest}"


def build_work_order(assessment: dict) -> WorkOrder:
    """Deterministically build a work order from an asset assessment dict."""
    asset_id = assessment["asset_id"]
    rul_p50 = float(assessment["rul_p50"])
    drivers = [d["sensor"] for d in assessment.get("top_drivers", [])]

    components, parts, skills = [], [], {"mechanical technician"}
    for sensor in drivers[:3]:
        comp, comp_parts = _COMPONENT_MAP.get(sensor, (f"{sensor} subsystem", ["inspection kit"]))
        components.append(comp)
        parts.extend(comp_parts)
        if "temperature" in comp or "coolant" in comp:
            skills.add("thermal systems specialist")
        if "pressure" in comp or "fuel" in comp:
            skills.add("hydraulics technician")

    priority = _priority_from_rul(rul_p50)
    ci = f"P10={assessment['rul_p10']}h / P50={rul_p50}h / P90={assessment['rul_p90']}h"
    summary = (
        f"Asset {asset_id} predicted to reach end-of-life in ~{rul_p50:.0f} hours "
        f"(anomaly score {assessment['anomaly_score']}). Degradation driven by "
        f"{', '.join(components)}."
    )
    action = (
        f"Schedule {priority} maintenance within {min(rul_p50, 48):.0f} hours. "
        f"Procure parts: {', '.join(sorted(set(parts)))}. "
        f"Dispatch: {', '.join(sorted(skills))}."
    )

    return WorkOrder(
        work_order_id=_wo_id(asset_id, rul_p50),
        asset_id=asset_id,
        priority=priority,
        predicted_failure_hours=round(rul_p50, 1),
        confidence_interval=ci,
        suspected_components=components,
        required_parts=sorted(set(parts)),
        technician_skills=sorted(skills),
        summary=summary,
        recommended_action=action,
        metadata={
            "physics_rul": assessment.get("physics_rul"),
            "physics_consistent": assessment.get("physics_consistent"),
            "health_score": assessment.get("health_score"),
        },
    )
