"""Real-time conservation tracking with sliding window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .graph import TensionGraph
from .laplacian import build_laplacian
from .eigen import eigendecompose
from .conservation import (
    conservation_ratios,
    spectral_gap,
    ConservationReport,
    ConservationRatio,
)
from .fingerprint import spectral_fingerprint


@dataclass
class Alert:
    """Alert from conservation drop detection."""
    timestamp: int  # observation index
    deviation: float
    message: str
    ratios: list[float]


class ConservationTracker:
    """Sliding-window tracker for real-time conservation monitoring.

    Maintains a window of recent observations and flags when conservation
    ratios deviate significantly from the baseline.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._observations: list[np.ndarray] = []
        self._baseline_ratios: Optional[list[float]] = None
        self._current_ratios: Optional[list[float]] = None
        self._observation_count: int = 0
        self._baseline_std: Optional[list[float]] = None
        self._ratio_history: list[list[float]] = []

    def feed(self, observation: np.ndarray) -> Optional[Alert]:
        """Feed a new observation. Returns an Alert if conservation drops.

        Args:
            observation: 1-D array of values (e.g., vertex attributes at this time step).

        Returns:
            Alert if anomaly detected, None otherwise.
        """
        observation = np.asarray(observation, dtype=np.float64)
        self._observations.append(observation)
        self._observation_count += 1

        # Keep sliding window
        if len(self._observations) > self.window_size:
            self._observations.pop(0)

        # Need at least 3 observations to build a graph
        if len(self._observations) < 3:
            return None

        # Build a transition graph from the window
        n = len(self._observations[0])
        transitions = []
        for i in range(len(self._observations) - 1):
            # Transition between consecutive observations
            src_idx = int(np.argmax(self._observations[i]))
            tgt_idx = int(np.argmax(self._observations[i + 1]))
            transitions.append((src_idx, tgt_idx))

        if not transitions:
            return None

        # Build graph and compute conservation
        graph = TensionGraph.build_from_transitions(transitions)
        if graph.vertex_count < 2:
            return None

        try:
            lap = build_laplacian(graph)
            eigen = eigendecompose(lap)

            # Use uniform attribute
            attr = np.ones(graph.vertex_count, dtype=np.float64)
            ratios = conservation_ratios(eigen, attr, "tracking")
            ratio_values = [r.ratio for r in ratios]
            self._current_ratios = ratio_values
            self._ratio_history.append(ratio_values)

            # Establish baseline
            if self._baseline_ratios is None and len(self._observations) >= min(10, self.window_size // 2):
                self._establish_baseline()

            # Check for anomalies
            if self._baseline_ratios is not None:
                return self._check_alert(ratio_values)
        except Exception:
            pass

        return None

    def _establish_baseline(self) -> None:
        """Set baseline from accumulated ratio history."""
        if len(self._ratio_history) < 3:
            return

        # Pad ratio histories to same length
        max_len = max(len(r) for r in self._ratio_history)
        padded = []
        for r in self._ratio_history:
            padded.append(r + [0.0] * (max_len - len(r)))

        arr = np.array(padded)
        self._baseline_ratios = arr.mean(axis=0).tolist()
        self._baseline_std = arr.std(axis=0).tolist()

    def _check_alert(self, current: list[float]) -> Optional[Alert]:
        """Check if current ratios deviate from baseline."""
        if self._baseline_ratios is None:
            return None

        max_len = max(len(current), len(self._baseline_ratios))
        cur = current + [0.0] * (max_len - len(current))
        base = self._baseline_ratios + [0.0] * (max_len - len(self._baseline_ratios))
        std = self._baseline_std + [1.0] * (max_len - len(self._baseline_std)) if self._baseline_std else [1.0] * max_len

        deviations = []
        for c, b, s in zip(cur, base, std):
            if s > 1e-12:
                deviations.append(abs(c - b) / s)
            else:
                deviations.append(0.0)

        max_dev = max(deviations) if deviations else 0.0
        threshold = 2.0  # 2 sigma

        if max_dev > threshold:
            return Alert(
                timestamp=self._observation_count,
                deviation=max_dev,
                message=f"Conservation drop detected: {max_dev:.2f}σ deviation",
                ratios=current,
            )
        return None

    def report(self) -> Optional[ConservationReport]:
        """Generate a full ConservationReport from the current window."""
        if len(self._observations) < 3:
            return None

        n = len(self._observations[0])
        transitions = []
        for i in range(len(self._observations) - 1):
            src_idx = int(np.argmax(self._observations[i]))
            tgt_idx = int(np.argmax(self._observations[i + 1]))
            transitions.append((src_idx, tgt_idx))

        if not transitions:
            return None

        graph = TensionGraph.build_from_transitions(transitions)
        if graph.vertex_count < 2:
            return None

        from .conservation import analyze
        return analyze(graph)

    @property
    def current_ratios(self) -> Optional[list[float]]:
        return self._current_ratios

    @property
    def baseline(self) -> Optional[list[float]]:
        return self._baseline_ratios

    def reset(self) -> None:
        self._observations = []
        self._baseline_ratios = None
        self._current_ratios = None
        self._baseline_std = None
        self._ratio_history = []
        self._observation_count = 0
