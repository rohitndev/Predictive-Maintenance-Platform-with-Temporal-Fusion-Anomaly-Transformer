"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single multivariate sensor reading (one cycle)."""

    op_settings: list[float] = Field(..., description="Operating-condition settings")
    sensors: list[float] = Field(..., description="Sensor channel values")


class AssetWindowRequest(BaseModel):
    """A window of sensor readings for one asset."""

    asset_id: str = Field(..., examples=["ENGINE-001"])
    readings: list[SensorReading] = Field(
        ..., description="Ordered sensor readings (most recent last)"
    )


class DriverWeight(BaseModel):
    sensor: str
    importance: float


class AssessmentResponse(BaseModel):
    asset_id: str
    rul_p10: float
    rul_p50: float
    rul_p90: float
    anomaly_score: float
    is_anomalous: bool
    physics_rul: float
    physics_agreement: float
    physics_consistent: bool
    health_score: float
    top_drivers: list[DriverWeight]
    needs_maintenance: bool


class WorkOrderResponse(BaseModel):
    created: bool
    work_order: dict | None = None
    reason: str | None = None


class BatchRequest(BaseModel):
    assets: list[AssetWindowRequest]


class BatchResponse(BaseModel):
    assessments: list[AssessmentResponse]
    work_orders_triggered: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    cloud: dict
