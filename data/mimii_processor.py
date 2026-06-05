"""MIMII Anomalous Sound Detection processor (optional acoustic modality).

The MIMII dataset (Zenodo) provides machine operating sound for anomaly
detection. This module extracts log-mel-spectrogram windows when audio files
are present. Audio dependencies (``librosa``/``soundfile``) are optional — if
they are missing or no audio is supplied, a synthetic spectrogram is returned
so the acoustic branch remains exercisable.

Dataset reference: MIMII (https://zenodo.org/record/3384388).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_MELS = 64
FRAMES = 50


def _synthetic_spectrogram(rng: np.random.Generator, anomalous: bool = False) -> np.ndarray:
    base = rng.normal(-40.0, 5.0, size=(N_MELS, FRAMES)).astype(np.float32)
    if anomalous:
        band = rng.integers(0, N_MELS - 8)
        base[band : band + 8, :] += rng.uniform(8.0, 18.0)
    return base


def extract_logmel(path: str | Path | None = None, anomalous: bool = False) -> np.ndarray:
    """Return a log-mel spectrogram for a MIMII clip (or a synthetic one)."""
    rng = np.random.default_rng(0)
    if path is None:
        return _synthetic_spectrogram(rng, anomalous)
    try:  # pragma: no cover - optional heavy deps
        import librosa

        y, sr = librosa.load(str(path), sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        logmel = librosa.power_to_db(mel)
        return logmel[:, :FRAMES].astype(np.float32)
    except Exception:
        return _synthetic_spectrogram(rng, anomalous)


if __name__ == "__main__":
    spec = extract_logmel(anomalous=True)
    print(f"Log-mel spectrogram shape: {spec.shape}")
