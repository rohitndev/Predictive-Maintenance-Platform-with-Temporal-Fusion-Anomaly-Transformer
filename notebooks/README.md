# Notebooks

Analysis and visualization for the dual-model platform.

- [`tft_analysis.py`](./tft_analysis.py) — TFT probabilistic RUL prediction
  (P10/P50/P90 confidence band), Anomaly Transformer detection example, and the
  physics-vs-ML comparison. Run with:

  ```bash
  python -m notebooks.tft_analysis
  ```

  Charts are written to [`../docs/`](../docs) when `matplotlib` is installed
  (`pip install matplotlib`); otherwise the metrics are printed.

These scripts double as the **Kaggle notebook** baseline:
*"Predicting Engine Remaining Useful Life with Temporal Fusion Transformer"* on
the NASA C-MAPSS turbofan dataset.
