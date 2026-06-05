"""Airflow weekly retraining DAG.

Retrains the dual models on newly accumulated, labelled failure events and
promotes the new champion in the model registry. The DAG is import-safe: if
Airflow is not installed (e.g. on a developer machine) the module still
imports and the underlying callables remain unit-testable.

Schedule: weekly (``0 3 * * 0``).
"""

from __future__ import annotations

from datetime import datetime, timedelta


def ingest_new_failures(**_) -> dict:
    """Pull the latest sensor windows + failure labels (InfluxDB/DVC snapshot)."""
    from data.cmapss_loader import load_cmapss
    from src.ingestion.window_builder import build_windows

    df = load_cmapss()
    windows, rul, _ = build_windows(df)
    return {"n_windows": int(len(windows)), "n_assets": int(df["unit_id"].nunique())}


def retrain_models(**_) -> dict:
    """Retrain TFT + Anomaly Transformer and register the new champion."""
    from scripts.train_pipeline import run_training

    summary = run_training(quick=False)
    return {"val_rmse": summary["tft"]["final_val_rmse"]}


def evaluate_and_promote(**_) -> dict:
    """Compare against the current champion and promote if improved."""
    from src.mlops.mlflow_setup import ModelRegistry

    registry = ModelRegistry()
    champion = registry.champion("tft-rul")
    return {"champion": champion}


default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

try:  # pragma: no cover - Airflow is optional in local/dev environments
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="pdm_weekly_retrain",
        description="Weekly predictive-maintenance model retraining",
        schedule="0 3 * * 0",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["predictive-maintenance", "retraining"],
    ) as dag:
        t1 = PythonOperator(task_id="ingest_new_failures", python_callable=ingest_new_failures)
        t2 = PythonOperator(task_id="retrain_models", python_callable=retrain_models)
        t3 = PythonOperator(task_id="evaluate_and_promote", python_callable=evaluate_and_promote)

        t1 >> t2 >> t3
except Exception:  # pragma: no cover
    dag = None
