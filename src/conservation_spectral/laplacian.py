"""Laplacian building — normalized and unnormalized variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

from .graph import TensionGraph


@dataclass
class Laplacian:
    """Computed Laplacian from a TensionGraph."""

    matrix: sparse.csr_matrix
    degree_matrix: sparse.csr_matrix
    weight_matrix: sparse.csr_matrix
    normalized: bool
    num_vertices: int

    def to_dense(self) -> np.ndarray:
        return self.matrix.toarray()


def build_laplacian(
    graph: TensionGraph,
    normalized: bool = True,
    laplacian_type: str = "symmetric_normalized",
) -> Laplacian:
    """Build a Laplacian from a TensionGraph.

    Args:
        graph: The tension graph.
        normalized: If True, use normalized Laplacian.
        laplacian_type: One of "unnormalized", "symmetric_normalized", "random_walk_normalized".

    Returns:
        Laplacian object with sparse matrices.
    """
    W = sparse.csr_matrix(graph.adjacency_matrix())
    n = graph.vertex_count

    # Degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sparse.diags(degrees, format="csr")

    if laplacian_type == "unnormalized" or not normalized:
        # L = D - W
        L = D - W
        is_normalized = False
    elif laplacian_type == "symmetric_normalized":
        # L = D^{-1/2} (D - W) D^{-1/2} = I - D^{-1/2} W D^{-1/2}
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csr")
        I = sparse.eye(n, format="csr")
        L = I - D_inv_sqrt @ W @ D_inv_sqrt
        is_normalized = True
    elif laplacian_type == "random_walk_normalized":
        # L = I - D^{-1} W
        d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
        D_inv = sparse.diags(d_inv, format="csr")
        I = sparse.eye(n, format="csr")
        L = I - D_inv @ W
        is_normalized = True
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    return Laplacian(
        matrix=L.tocsr(),
        degree_matrix=D,
        weight_matrix=W,
        normalized=is_normalized,
        num_vertices=n,
    )
