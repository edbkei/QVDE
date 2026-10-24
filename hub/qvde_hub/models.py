"""
Detection models used by the Cognition module.

The Hub reuses the custom LSTM of src/custom_lstm_persistence.py. When no
trained LSTM file is available (e.g. a fresh simulation setup), a transparent
SMV-peak heuristic keeps the pipeline running; its version string makes the
substitution visible in every uplink record.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))


class SMVHeuristic:
    """P(fall) from the peak signal-magnitude vector of the acceleration (in g)."""
    version = "heuristic-smv-v1"
    sequence_length = 10

    def __init__(self, g: float = 9.81, low_g: float = 1.2, high_g: float = 2.5):
        self.g, self.low, self.high = g, low_g, high_g

    def predict_proba(self, seq: np.ndarray) -> float:
        smv = np.sqrt((seq[:, :3] ** 2).sum(axis=1)) / self.g
        x = (smv.max() - self.low) / (self.high - self.low)
        return float(min(1.0, max(0.0, x)))


class LSTMModel:
    def __init__(self, path: str):
        from custom_lstm_persistence import SimpleLSTM  # noqa: WPS433 (runtime import)
        self._m = SimpleLSTM.load(path, format="pickle")
        self.sequence_length = self._m.sequence_length
        match = re.search(r"v\d+\.\d+\.\d+", os.path.basename(path))
        self.version = f"lstm-{match.group(0)}" if match else f"lstm-{_sha(path)[:8]}"

    def predict_proba(self, seq: np.ndarray) -> float:
        return float(self._m.predict(seq)) / 100.0  # SimpleLSTM returns a percentage


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_lstm(model_dir: str) -> str | None:
    def key(p: str):
        m = re.search(r"v(\d+)\.(\d+)\.(\d+)_(\d{8}_\d{6})", p)
        return (tuple(map(int, m.groups()[:3])), m.group(4)) if m else ((0, 0, 0), "")
    files = glob.glob(os.path.join(model_dir, "lstm_v*.pkl"))
    return max(files, key=key) if files else None


def load_detection_model(path: str | None = None, model_dir: str | None = None):
    path = path or (latest_lstm(model_dir) if model_dir else None)
    if path and os.path.exists(path):
        try:
            return LSTMModel(path)
        except Exception as exc:  # corrupt file or missing dependency
            print(f"[models] could not load {path}: {exc}; using heuristic")
    return SMVHeuristic()


__all__ = ["SMVHeuristic", "LSTMModel", "load_detection_model", "latest_lstm"]
