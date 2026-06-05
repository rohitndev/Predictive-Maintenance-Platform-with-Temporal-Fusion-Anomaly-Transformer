"""FastAPI serving layer for the Predictive Maintenance Platform.

Exposes real-time RUL prediction, anomaly scoring, asset-health assessment,
batch scoring, and the CMMS work-order trigger. Designed to run locally with
Uvicorn or behind AWS Lambda + API Gateway (see ``deployment/lambda_handler``).

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

import src
from api.schemas import (
    AssessmentResponse,
    AssetWindowRequest,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    WorkOrderResponse,
)
from src.config import CLOUD, SENSOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = src.__version__

_STATE: dict = {"ensemble": None, "agent": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_models()
    yield


app = FastAPI(
    title="Predictive Maintenance Platform with Temporal Fusion & Anomaly Transformer",
    description=(
        "Dual-model predictive maintenance API: Temporal Fusion Transformer for "
        "probabilistic RUL prediction and Anomaly Transformer for point/contextual "
        "anomaly detection, with physics-based digital-twin validation and an "
        "automated CMMS work-order agent."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _window_from_request(req: AssetWindowRequest) -> np.ndarray:
    rows = []
    for r in req.readings:
        row = list(r.op_settings) + list(r.sensors)
        if len(row) != SENSOR.n_op_settings + SENSOR.n_sensors:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Each reading needs {SENSOR.n_op_settings} op_settings + "
                    f"{SENSOR.n_sensors} sensors."
                ),
            )
        rows.append(row)
    arr = np.asarray(rows, dtype=np.float32)
    if len(arr) < SENSOR.window_size:
        pad = np.repeat(arr[:1], SENSOR.window_size - len(arr), axis=0)
        arr = np.vstack([pad, arr])
    return arr[-SENSOR.window_size :]


def _ensure_models() -> None:
    """Lazy-load the ensemble; auto-train a quick model if none exists yet."""
    if _STATE["ensemble"] is not None:
        return
    from src.models.ensemble import load_ensemble

    try:
        _STATE["ensemble"] = load_ensemble()
        logger.info("Loaded trained models from artifacts.")
    except Exception as exc:
        logger.warning("No trained artifacts (%s); training a quick model...", exc)
        from scripts.train_pipeline import run_training

        run_training(quick=True)
        _STATE["ensemble"] = load_ensemble()

    from src.agent.cmms_agent import CMMSAgent

    _STATE["agent"] = CMMSAgent()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=_STATE["ensemble"] is not None,
        version=__version__,
        cloud={
            "influxdb": CLOUD.influx_enabled,
            "aws_s3": CLOUD.aws_enabled,
            "groq_llm": CLOUD.groq_enabled,
        },
    )


@app.post("/predict/rul", response_model=AssessmentResponse)
def predict_rul(req: AssetWindowRequest) -> AssessmentResponse:
    """Predict probabilistic RUL + anomaly + physics validation for one asset."""
    _ensure_models()
    window = _window_from_request(req)
    assessment = _STATE["ensemble"].assess(req.asset_id, window)
    return AssessmentResponse(**assessment.to_dict())


@app.post("/predict/anomaly")
def predict_anomaly(req: AssetWindowRequest) -> dict:
    """Return only the anomaly score for one asset window."""
    _ensure_models()
    window = _window_from_request(req)
    assessment = _STATE["ensemble"].assess(req.asset_id, window)
    return {
        "asset_id": req.asset_id,
        "anomaly_score": assessment.anomaly_score,
        "is_anomalous": assessment.is_anomalous,
    }


@app.post("/maintenance/trigger", response_model=WorkOrderResponse)
def trigger_maintenance(req: AssetWindowRequest, force: bool = False) -> WorkOrderResponse:
    """Assess an asset and, if needed, generate + file a CMMS work order."""
    _ensure_models()
    window = _window_from_request(req)
    assessment = _STATE["ensemble"].assess(req.asset_id, window).to_dict()
    confirmation = _STATE["agent"].process_assessment(assessment, force=force)
    if confirmation is None:
        return WorkOrderResponse(
            created=False, reason="Asset healthy; no maintenance required."
        )
    return WorkOrderResponse(created=True, work_order=confirmation)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Score multiple assets and count maintenance triggers."""
    _ensure_models()
    assessments, triggered = [], 0
    for asset in req.assets:
        window = _window_from_request(asset)
        assessment = _STATE["ensemble"].assess(asset.asset_id, window)
        assessments.append(AssessmentResponse(**assessment.to_dict()))
        if assessment.needs_maintenance:
            triggered += 1
    return BatchResponse(assessments=assessments, work_orders_triggered=triggered)


@app.get("/")
def root() -> dict:
    return {
        "service": "Predictive Maintenance Platform with Temporal Fusion & Anomaly Transformer",
        "docs": "/docs",
        "health": "/health",
    }
