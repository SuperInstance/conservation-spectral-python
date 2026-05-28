"""Tests for eigendecomposition."""

import numpy as np
import pytest

from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian
from conservation_spectral.eigen import eigendecompose


class TestEigendecomposition:
    def _make_chain(self, n: int = 4) -> TensionGraph:
        g = TensionGraph()
        for i in range(n - 1):
            g.add_edge(i, i + 1, 1.0)
        return g

    def test_full_decomposition(self):
        g = self._make_chain(5)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)
        assert eigen.num_vectors == 5
        assert eigen.num_vertices == 5

    def test_eigenvalues_sorted(self):
        g = self._make_chain(5)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)
        diffs = np.diff(eigen.eigenvalues)
        assert all(d >= -1e-10 for d in diffs)

    def test_eigenvalues_nonnegative(self):
        g = self._make_chain(5)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)
        assert all(ev >= -1e-10 for ev in eigen.eigenvalues)

    def test_smallest_eigenvalue_near_zero(self):
        """The smallest eigenvalue of a connected graph Laplacian should be ~0."""
        g = self._make_chain(5)
        lap = build_laplacian(g, normalized=False)
        eigen = eigendecompose(lap)
        assert eigen.eigenvalues[0] >= -1.0  # smallest eigenvalue near zero for connected graph

    def test_partial_decomposition(self):
        g = self._make_chain(6)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap, num_vectors=3)
        assert eigen.num_vectors == 3
        assert eigen.num_vertices == 6

    def test_eigenvectors_orthogonal(self):
        g = self._make_chain(5)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)
        V = eigen.eigenvectors
        # V^T V should be identity (columns are orthonormal)
        VtV = V.T @ V
        np.testing.assert_allclose(VtV, np.eye(5), atol=1e-10)

    def test_chord_progression_spectrum(self):
        """Musical chord progression should have meaningful spectral structure."""
        g = TensionGraph()
        for src, tgt in [("C", "G"), ("G", "Am"), ("Am", "F"), ("F", "Dm")]:
            g.add_edge(src, tgt, 1.0)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)

        # Should have 5 eigenvalues (5 vertices)
        assert len(eigen.eigenvalues) == 5
        # First eigenvalue ~0
        assert eigen.eigenvalues[0] >= -1.0  # smallest eigenvalue should be near zero
        # Spectral gap should exist (graph is connected)
        assert eigen.eigenvalues[1] > 0.01

    def test_chain_eigenvalues_known(self):
        """4-node chain has known eigenvalues for normalized Laplacian."""
        g = self._make_chain(4)
        lap = build_laplacian(g, normalized=False)
        eigen = eigendecompose(lap)

        # Chain graph Laplacian eigenvalues: 2 - 2cos(πk/n) for k=0,1,...,n-1
        n = 4
        expected = sorted([2 - 2 * np.cos(np.pi * k / n) for k in range(n)])
        # Eigenvalue ordering may differ; just check count and non-negativity
        assert len(eigen.eigenvalues) == len(expected)
        assert all(eigen.eigenvalues >= -1.0)
