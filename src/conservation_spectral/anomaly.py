"""Anomaly detection and correction suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .graph import TensionGraph
from .eigen import EigenDecomposition


class AnomalyType(str, Enum):
    CONSERVATION_VIOLATION = "conservation_violation"
    STRUCTURAL_BREAK = "structural_break"
    SPECTRAL_OUTLIER = "spectral_outlier"
    TRANSITION_ANOMALY = "transition_anomaly"


@dataclass
class Anomaly:
    vertex_id: int
    eigenvector_index: int
    deviation: float
    anomaly_type: AnomalyType
    description: str


@dataclass
class Fix:
    edge: Optional[tuple[int, int]] = None
    vertex: Optional[int] = None
    suggested_weight: Optional[float] = None
    description: str = ""
    confidence: float = 0.0


def detect_anomalies(
    graph: TensionGraph,
    eigen: Optional[EigenDecomposition] = None,
    threshold: float = 2.0,
) -> list[Anomaly]:
    """Detect anomalies in a graph based on spectral analysis.

    Looks for vertices where the eigenvector components deviate significantly
    from the mean, indicating conservation violations or structural breaks.

    Args:
        graph: The tension graph.
        eigen: Pre-computed eigendecomposition. If None, computes it.
        threshold: Number of standard deviations for flagging.

    Returns:
        List of detected Anomalies.
    """
    if graph.vertex_count < 3:
        return []

    if eigen is None:
        from .laplacian import build_laplacian
        from .eigen import eigendecompose
        lap = build_laplacian(graph)
        eigen = eigendecompose(lap)

    anomalies: list[Anomaly] = []

    # Check each eigenvector for outliers
    for k in range(min(eigen.num_vectors, graph.vertex_count)):
        vec = eigen.eigenvectors[:, k]
        mean = np.mean(vec)
        std = np.std(vec)

        if std < 1e-15:
            continue

        z_scores = np.abs((vec - mean) / std)

        for i in range(graph.vertex_count):
            if z_scores[i] > threshold:
                # Determine anomaly type
                if k == 0:
                    atype = AnomalyType.STRUCTURAL_BREAK
                elif z_scores[i] > 3.0:
                    atype = AnomalyType.SPECTRAL_OUTLIER
                else:
                    atype = AnomalyType.CONSERVATION_VIOLATION

                anomalies.append(Anomaly(
                    vertex_id=i,
                    eigenvector_index=k,
                    deviation=float(z_scores[i]),
                    anomaly_type=atype,
                    description=(
                        f"Vertex {graph.vertices[i]} (idx={i}) deviates {z_scores[i]:.2f}σ "
                        f"from mean in eigenvector {k} (eigenvalue={eigen.eigenvalues[k]:.4f})"
                    ),
                ))

    # Deduplicate: keep highest deviation per vertex
    best: dict[int, Anomaly] = {}
    for a in anomalies:
        if a.vertex_id not in best or a.deviation > best[a.vertex_id].deviation:
            best[a.vertex_id] = a

    return sorted(best.values(), key=lambda a: -a.deviation)


def suggest_correction(
    graph: TensionGraph,
    anomaly: Anomaly,
) -> list[Fix]:
    """Suggest corrections for a detected anomaly.

    Heuristic: look at edges incident to the anomalous vertex and
    suggest weight adjustments that would bring the vertex closer to the mean.

    Args:
        graph: The tension graph.
        anomaly: The detected anomaly.

    Returns:
        List of suggested Fixes.
    """
    fixes: list[Fix] = []
    vi = anomaly.vertex_id
    W = graph.adjacency_matrix()

    # Get incident edges
    for j in range(graph.vertex_count):
        if W[vi, j] > 0 or W[j, vi] > 0:
            # Suggest reducing the weight to normalize the vertex
            current_weight = max(W[vi, j], W[j, vi])
            suggested = current_weight * 0.8  # reduce by 20%

            fixes.append(Fix(
                edge=(vi, j),
                vertex=vi,
                suggested_weight=suggested,
                description=(
                    f"Reduce edge weight between vertices {graph.vertices[vi]} and "
                    f"{graph.vertices[j]} from {current_weight:.3f} to {suggested:.3f} "
                    f"to reduce spectral deviation"
                ),
                confidence=0.6,
            ))

    if not fixes:
        # Suggest adding edges if the vertex is isolated
        fixes.append(Fix(
            vertex=vi,
            description=(
                f"Vertex {graph.vertices[vi]} has no incident edges. "
                f"Consider adding transitions to reduce isolation."
            ),
            confidence=0.3,
        ))

    return fixes
