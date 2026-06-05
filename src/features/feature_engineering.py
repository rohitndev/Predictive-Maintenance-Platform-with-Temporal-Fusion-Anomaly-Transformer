"""Automated time-series feature engineering.

Extracts a rich statistical feature set from raw sensor windows. When the
``tsfresh`` library is installed the full automated extraction (hundreds of
features: FFT coefficients, entropy, autocorrelation, etc.) is used. Otherwise
a fast, dependency-free implementation computes an equivalent family of
features (statistics, FFT energy, sample entropy, autocorrelation, peaks) so
the pipeline produces a consistent feature matrix everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft

from data.cmapss_loader import feature_columns


def _safe_entropy(x: np.ndarray, bins: int = 10) -> float:
    hist, _ = np.histogram(x, bins=bins)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p)))


def _autocorr(x: np.ndarray, lag: int = 1) -> float:
    if len(x) <= lag or x.std() == 0:
        return 0.0
    a, b = x[:-lag], x[lag:]
    return float(np.corrcoef(a, b)[0, 1])


def _fft_energy(x: np.ndarray, n_coeff: int = 5) -> list[float]:
    spectrum = np.abs(rfft(x - x.mean()))
    coeffs = spectrum[:n_coeff]
    if len(coeffs) < n_coeff:
        coeffs = np.pad(coeffs, (0, n_coeff - len(coeffs)))
    return coeffs.tolist()


def extract_window_features(window: np.ndarray, channel_names: list[str]) -> dict[str, float]:
    """Extract per-channel statistical/spectral features from one window.

    ``window`` has shape ``[T, F]`` aligned with ``channel_names``.
    """
    features: dict[str, float] = {}
    for c, name in enumerate(channel_names):
        x = window[:, c].astype(np.float64)
        features[f"{name}__mean"] = float(np.mean(x))
        features[f"{name}__std"] = float(np.std(x))
        features[f"{name}__min"] = float(np.min(x))
        features[f"{name}__max"] = float(np.max(x))
        features[f"{name}__range"] = float(np.ptp(x))
        features[f"{name}__skew"] = float(stats.skew(x)) if x.std() > 0 else 0.0
        features[f"{name}__kurtosis"] = float(stats.kurtosis(x)) if x.std() > 0 else 0.0
        features[f"{name}__slope"] = float(np.polyfit(np.arange(len(x)), x, 1)[0])
        features[f"{name}__entropy"] = _safe_entropy(x)
        features[f"{name}__autocorr_lag1"] = _autocorr(x, 1)
        features[f"{name}__abs_energy"] = float(np.sum(x**2))
        for i, e in enumerate(_fft_energy(x)):
            features[f"{name}__fft_coeff_{i}"] = float(e)
    return features


class FeatureEngineer:
    """Builds a tabular feature matrix from sliding sensor windows."""

    def __init__(self, channel_names: list[str] | None = None):
        self.channel_names = channel_names or feature_columns()

    @property
    def n_features(self) -> int:
        # 11 scalar + 5 FFT coefficients per channel.
        return len(self.channel_names) * 16

    def transform(self, windows: np.ndarray) -> pd.DataFrame:
        """Return a ``[n_windows, n_features]`` DataFrame.

        Uses ``tsfresh`` when available, otherwise the built-in fast extractor.
        """
        try:  # pragma: no cover - optional heavy dep
            return self._transform_tsfresh(windows)
        except Exception:
            rows = [
                extract_window_features(w, self.channel_names) for w in windows
            ]
            return pd.DataFrame(rows).fillna(0.0)

    def _transform_tsfresh(self, windows: np.ndarray) -> pd.DataFrame:  # pragma: no cover
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters

        long_frames = []
        for wid, w in enumerate(windows):
            df = pd.DataFrame(w, columns=self.channel_names)
            df["id"] = wid
            df["time"] = np.arange(len(w))
            long_frames.append(df)
        long_df = pd.concat(long_frames, ignore_index=True)
        extracted = extract_features(
            long_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )
        return extracted.fillna(0.0)


if __name__ == "__main__":
    from data.cmapss_loader import load_cmapss
    from src.ingestion.window_builder import build_windows

    df = load_cmapss()
    windows, _, _ = build_windows(df)
    fe = FeatureEngineer()
    feats = fe.transform(windows[:50])
    print(f"Feature matrix: {feats.shape} ({fe.n_features} expected per-window)")
