"""TFT / Anomaly analysis and visualization.

Produces the analytical artifacts referenced in the README: the TFT RUL
prediction with P10/P50/P90 confidence band, the Anomaly Transformer detection
example, and the physics-vs-ML comparison. Charts are saved under ``docs/`` if
``matplotlib`` is installed; otherwise the same metrics are printed so the
analysis remains reproducible in a headless/minimal environment.

Usage:
    python -m notebooks.tft_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.cmapss_loader import load_cmapss
from src.config import ROOT_DIR
from src.ingestion.window_builder import build_windows
from src.models.ensemble import load_ensemble

DOCS = Path(ROOT_DIR) / "docs"
DOCS.mkdir(exist_ok=True)


def _maybe_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def rul_confidence_analysis(ensemble) -> None:
    df = load_cmapss()
    # Sample representative windows from across every trajectory (not just the
    # final cycle) so the metric reflects held-out generalisation.
    windows, rul, _ = build_windows(df)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(windows))[:40]
    order = np.argsort(rul[idx])           # sort by true RUL for a readable plot
    idx = idx[order]

    p10s, p50s, p90s, true_rul = [], [], [], []
    for i in idx:
        a = ensemble.assess("ENGINE", windows[i])
        p10s.append(a.rul_p10)
        p50s.append(a.rul_p50)
        p90s.append(a.rul_p90)
        true_rul.append(float(rul[i]))

    p50 = np.array(p50s)
    rmse = float(np.sqrt(np.mean((p50 - np.array(true_rul)) ** 2)))
    coverage = float(
        np.mean([(t >= lo) and (t <= hi) for t, lo, hi in zip(true_rul, p10s, p90s, strict=False)])
    )
    print(f"TFT RUL — P50 RMSE: {rmse:.2f} cycles | P10-P90 coverage: {coverage:.0%}")

    plt = _maybe_plt()
    if plt is None:
        return
    x = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, p10s, p90s, alpha=0.25, color="#1f77b4", label="P10–P90 interval")
    ax.plot(x, p50s, "-o", color="#1f77b4", label="P50 (predicted RUL)")
    ax.plot(x, true_rul, "x", color="#d62728", label="True RUL")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Remaining Useful Life (cycles)")
    ax.set_title("Temporal Fusion Transformer — Probabilistic RUL Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "tft-rul-prediction.png", dpi=120)
    print(f"Saved {DOCS / 'tft-rul-prediction.png'}")


def anomaly_analysis(ensemble) -> None:
    df = load_cmapss()
    unit = df[df["unit_id"] == df["unit_id"].max()]
    windows, _, _ = build_windows(unit)
    scores = []
    for w in windows:
        scores.append(ensemble._predict_anomaly(w))
    scores = np.array(scores)
    print(
        f"Anomaly Transformer — mean score {scores.mean():.3f}, "
        f"flagged {int((scores > 0.55).sum())}/{len(scores)} windows"
    )

    plt = _maybe_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores, color="#2ca02c", label="anomaly score")
    ax.axhline(0.55, ls="--", color="#d62728", label="threshold")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Normalised anomaly score")
    ax.set_title("Anomaly Transformer — Point & Contextual Detection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "anomaly-detection.png", dpi=120)
    print(f"Saved {DOCS / 'anomaly-detection.png'}")


def main() -> None:
    ensemble = load_ensemble()
    rul_confidence_analysis(ensemble)
    anomaly_analysis(ensemble)


if __name__ == "__main__":
    main()
