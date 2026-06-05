"""LangChain CMMS integration agent.

Reads an RUL/anomaly assessment, generates a natural-language maintenance work
order using an LLM (Groq / Mixtral via LangChain), and submits it to the
SAP PM / ServiceMax CMMS. When no Groq API key is configured (or LangChain is
not installed) the agent transparently falls back to the deterministic
template generator, so the end-to-end flow always produces and files a valid
work order.
"""

from __future__ import annotations

import json
import logging

from src.agent.sap_pm_api import SAPMaintenanceClient
from src.agent.work_order import WorkOrder, build_work_order
from src.config import CLOUD, TRIGGER

logger = logging.getLogger(__name__)

_LLM_PROMPT = """You are an industrial maintenance planning agent for a CMMS.
Given the predictive-maintenance assessment below, write a concise, actionable
maintenance work order.

Assessment (JSON):
{assessment}

Respond with a short professional summary (2-3 sentences) describing the
predicted failure, the urgency, the suspected components, and the recommended
action. Do not invent part numbers that are not provided."""


class CMMSAgent:
    """Orchestrates work-order generation and CMMS submission."""

    def __init__(self) -> None:
        self.cmms = SAPMaintenanceClient()
        self._llm = self._init_llm()

    def _init_llm(self):
        if not CLOUD.groq_enabled:
            logger.info("No Groq API key; CMMS agent uses template generator.")
            return None
        try:  # pragma: no cover - requires Groq API
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                api_key=CLOUD.groq_api_key,
                model=CLOUD.groq_model,
                temperature=0.2,
            )
            logger.info("CMMS agent using Groq model %s", CLOUD.groq_model)
            return llm
        except Exception as exc:  # pragma: no cover
            logger.warning("Groq/LangChain unavailable (%s); using template.", exc)
            return None

    def _llm_summary(self, assessment: dict) -> str | None:
        if self._llm is None:
            return None
        try:  # pragma: no cover - requires Groq API
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(_LLM_PROMPT)
            chain = prompt | self._llm
            result = chain.invoke({"assessment": json.dumps(assessment, indent=2)})
            return getattr(result, "content", str(result)).strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM generation failed (%s); falling back.", exc)
            return None

    def generate_work_order(self, assessment: dict) -> WorkOrder:
        """Build a work order, enriching the summary with the LLM if available."""
        work_order = build_work_order(assessment)
        llm_text = self._llm_summary(assessment)
        if llm_text:
            work_order.summary = llm_text
            work_order.generated_by = f"groq:{CLOUD.groq_model}"
        return work_order

    def process_assessment(self, assessment: dict, force: bool = False) -> dict | None:
        """Full agentic flow: decide, generate, and file a work order.

        Returns the CMMS confirmation, or ``None`` if no maintenance is needed.
        """
        needs = assessment.get("needs_maintenance", False)
        if not needs and not force:
            logger.info(
                "Asset %s healthy (RUL P50=%.1fh, anomaly=%.3f); no work order.",
                assessment.get("asset_id"),
                assessment.get("rul_p50", 0.0),
                assessment.get("anomaly_score", 0.0),
            )
            return None

        work_order = self.generate_work_order(assessment)
        confirmation = self.cmms.create_work_order(work_order.to_dict())
        logger.info(
            "Filed %s for asset %s (%s) %.0fh before predicted failure.",
            work_order.work_order_id,
            work_order.asset_id,
            work_order.priority,
            TRIGGER.rul_alert_hours,
        )
        return confirmation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = {
        "asset_id": "ENGINE-007",
        "rul_p10": 8.0,
        "rul_p50": 22.0,
        "rul_p90": 40.0,
        "anomaly_score": 0.78,
        "is_anomalous": True,
        "physics_rul": 25.0,
        "physics_consistent": True,
        "health_score": 0.18,
        "needs_maintenance": True,
        "top_drivers": [
            {"sensor": "sensor_3", "importance": 0.21},
            {"sensor": "sensor_12", "importance": 0.18},
            {"sensor": "sensor_20", "importance": 0.11},
        ],
    }
    agent = CMMSAgent()
    print(json.dumps(agent.process_assessment(demo), indent=2))
