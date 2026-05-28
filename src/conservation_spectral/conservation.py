"""Conservation analysis — ratios, spectral gap, Cheeger constant, full reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .graph import TensionGraph
from .laplacian import Laplacian, build_laplacian
from .eigen import EigenDecomposition, eigendecompose



@dataclass
class ConservationRatio:
    eigenvector_index: int
    eigenvalue: float
    ratio: float
    attribute_name: str


@dataclass
class ConservationReport:
    ratios: list[ConservationRatio]
    anomalies: list  # List[Anomaly] from anomaly module
    spectral_gap: float
    cheeger_constant: float
    fingerprint: SpectralFingerprint

    def to_dict(self) -> dict:
        return {
            "spectral_gap": self.spectral_gap,
            "cheeger_constant": self.cheeger_constant,
            "ratios": [
                {
                    "eigenvector_index": r.eigenvector_index,
                    "eigenvalue": r.eigenvalue,
                    "ratio": r.ratio,
                    "attribute_name": r.attribute_name,
                }
                for r in self.ratios
            ],
            "fingerprint": {
                "spectral_entropy": self.fingerprint.spectral_entropy,
                "effective_dimension": self.fingerprint.effective_dimension,
                "gap_profile": self.fingerprint.gap_profile.tolist()
                    if isinstance(self.fingerprint.gap_profile, np.ndarray)
                    else self.fingerprint.gap_profile,
            },
            "num_anomalies": len(self.anomalies),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def conservation_ratio(
    eigen: EigenDecomposition,
    attribute: np.ndarray,
    eigenvector_index: int,
    attribute_name: str = "default",
) -> float:
    """Compute conservation ratio of an attribute along the k-th eigenvector.

    CR(k) = Var(gradient of attribute projected onto eigenvector_k)
    Low ratio = attribute is well-conserved in this mode.
    """
    attribute = np.asarray(attribute, dtype=np.float64)
    phi = eigen.eigenvectors[:, eigenvector_index]

    # Project attribute onto eigenvector
    projection = phi * attribute

    # Compute gradient (finite differences)
    gradient = np.diff(projection)

    if len(gradient) == 0:
        return float("inf")

    # Variance of gradient
    mean = np.mean(gradient)
    var = np.mean((gradient - mean) ** 2)
    return float(var)


def conservation_ratios(
    eigen: EigenDecomposition,
    attribute: np.ndarray,
    attribute_name: str = "default",
) -> list[ConservationRatio]:
    """Compute conservation ratios for all eigenvectors."""
    return [
        ConservationRatio(
            eigenvector_index=k,
            eigenvalue=float(eigen.eigenvalues[k]),
            ratio=conservation_ratio(eigen, attribute, k, attribute_name),
            attribute_name=attribute_name,
        )
        for k in range(eigen.num_vectors)
    ]


def spectral_gap(eigenvalues: np.ndarray) -> float:
    """Compute the spectral gap: largest gap between consecutive eigenvalues."""
    if len(eigenvalues) < 2:
        return 0.0
    gaps = np.diff(eigenvalues)
    # Exclude the trivial zero eigenvalue gap (index 0)
    if len(gaps) > 1:
        return float(np.max(gaps[1:])) if len(gaps) > 1 else float(gaps[0])
    return float(gaps[0])


def cheeger_constant(laplacian: Laplacian, fiedler_vector: Optional[np.ndarray] = None) -> float:
    """Approximate the Cheeger constant from the Fiedler vector.

    Uses the spectral approximation: h ≈ λ₂ / 2
    If fiedler_vector is provided, uses the sweep cut method.
    """
    if fiedler_vector is not None:
        # Sweep cut: sort vertices by Fiedler vector, find minimum cut ratio
        n = len(fiedler_vector)
        order = np.argsort(fiedler_vector)
        sorted_fiedler = fiedler_vector[order]

        W = laplacian.weight_matrix.toarray()
        degrees = np.array(W.sum(axis=1)).flatten()
        total_vol = degrees.sum()

        best_h = float("inf")
        vol_s = 0.0
        cut = 0.0

        in_s = np.zeros(n, dtype=bool)
        for i in range(n - 1):
            vi = order[i]
            in_s[vi] = True
            vol_s += degrees[vi]

            # Update cut
            for j in range(n):
                if in_s[j] != in_s.get(order[0]) if False else False:
                    pass
            # Simpler: recompute cut each step
            cut = 0.0
            for si in range(n):
                if in_s[si]:
                    for j in range(n):
                        if not in_s[j]:
                            cut += W[si, j]

            vol_complement = total_vol - vol_s
            if vol_s > 0 and vol_complement > 0:
                h = cut / min(vol_s, vol_complement)
                best_h = min(best_h, h)

        return float(best_h) if best_h != float("inf") else 0.0
    else:
        # Fallback: use eigenvalue-based approximation
        eigen = eigendecompose(laplacian, num_vectors=2)
        if len(eigen.eigenvalues) >= 2:
            return float(eigen.eigenvalues[1]) / 2.0
        return 0.0


def analyze(
    graph: TensionGraph,
    attribute_name: str = "default",
    attribute: Optional[np.ndarray] = None,
    laplacian_type: str = "symmetric_normalized",
) -> ConservationReport:
    from .fingerprint import spectral_fingerprint
    """One-call convenience: graph → Laplacian → eigendecomposition → report.

    Args:
        graph: TensionGraph to analyze.
        attribute_name: Name of the vertex attribute to analyze.
        attribute: Explicit attribute array. If None, looks up graph attribute.
        laplacian_type: Type of Laplacian to build.

    Returns:
        ConservationReport with full analysis.
    """
    if attribute is None:
        try:
            attribute = graph.get_attribute(attribute_name)
        except KeyError:
            # Default: use vertex indices as attribute
            attribute = np.arange(graph.vertex_count, dtype=np.float64)

    lap = build_laplacian(graph, normalized=True, laplacian_type=laplacian_type)
    eigen = eigendecompose(lap, laplacian_type=laplacian_type)

    ratios = conservation_ratios(eigen, attribute, attribute_name)
    fp = spectral_fingerprint(eigen, ratios)

    sg = spectral_gap(eigen.eigenvalues)

    # Cheeger constant from Fiedler vector
    if eigen.num_vectors >= 2:
        ch = float(eigen.eigenvalues[1]) / 2.0
    else:
        ch = 0.0

    # Detect anomalies
    from .anomaly import detect_anomalies
    anomalies = detect_anomalies(graph, eigen, threshold=2.0)

    return ConservationReport(
        ratios=ratios,
        anomalies=anomalies,
        spectral_gap=sg,
        cheeger_constant=ch,
        fingerprint=fp,
    )
