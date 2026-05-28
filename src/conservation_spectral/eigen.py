"""Eigendecomposition — sparse Lanczos via scipy.sparse.linalg."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .laplacian import Laplacian


@dataclass
class EigenDecomposition:
    """Result of eigendecomposition of a Laplacian."""

    eigenvalues: np.ndarray     # (k,) sorted ascending
    eigenvectors: np.ndarray    # (n, k) columns = eigenvectors
    laplacian_type: str

    @property
    def num_vectors(self) -> int:
        return len(self.eigenvalues)

    @property
    def num_vertices(self) -> int:
        return self.eigenvectors.shape[0]


def eigendecompose(
    laplacian: Laplacian,
    num_vectors: int = 0,
    laplacian_type: str = "symmetric_normalized",
) -> EigenDecomposition:
    """Compute eigendecomposition of a Laplacian.

    Uses scipy.sparse.linalg.eigsh (Lanczos) for the k smallest eigenvalues.
    These are the conservation-relevant modes.

    Args:
        laplacian: Laplacian from build_laplacian().
        num_vectors: Number of eigenvectors. 0 = all (falls back to dense).
        laplacian_type: Label for the type used.

    Returns:
        EigenDecomposition with eigenvalues sorted ascending.
    """
    L = laplacian.matrix
    n = laplacian.num_vertices

    if num_vectors == 0 or num_vectors >= n:
        # Full decomposition via dense
        L_dense = L.toarray()
        # Symmetrize to avoid numerical issues
        L_dense = (L_dense + L_dense.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        return EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            laplacian_type=laplacian_type,
        )
    else:
        k = min(num_vectors, n - 2)  # eigsh requires k < n
        # sigma=0 finds smallest eigenvalues via shift-invert mode
        # But for simplicity, use which="SM" (smallest magnitude)
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        except Exception:
            # Fallback to dense if sparse fails (e.g., singular)
            L_dense = L.toarray()
            L_dense = (L_dense + L_dense.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

        # Sort ascending
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            laplacian_type=laplacian_type,
        )
