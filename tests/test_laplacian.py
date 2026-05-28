"""Tests for Laplacian building."""

import numpy as np
import pytest
from scipy import sparse

from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian


class TestLaplacian:
    def _make_chain(self, n: int = 4) -> TensionGraph:
        g = TensionGraph()
        for i in range(n - 1):
            g.add_edge(i, i + 1, 1.0)
        return g

    def test_unnormalized_laplacian(self):
        g = self._make_chain(4)
        lap = build_laplacian(g, normalized=False, laplacian_type="unnormalized")
        assert lap.num_vertices == 4
        assert not lap.normalized

        L = lap.to_dense()
        # Chain graph: D = diag([1,2,2,1]), W has edges 0-1,1-2,2-3
        # L = D - W
        assert L[0, 0] == pytest.approx(1.0)
        assert L[1, 1] >= 1.0  # degree at least 1
        assert L[0, 1] == pytest.approx(-1.0)
        assert L[2, 3] == pytest.approx(-1.0)

    def test_normalized_laplacian(self):
        g = self._make_chain(4)
        lap = build_laplacian(g, normalized=True, laplacian_type="symmetric_normalized")
        assert lap.normalized

        L = lap.to_dense()
        # Diagonal should be 1.0 for normalized
        assert L[0, 0] == pytest.approx(1.0)
        assert L[1, 1] == pytest.approx(1.0)

    def test_row_sums_zero(self):
        """Laplacian rows should sum to ~0."""
        g = self._make_chain(5)
        lap = build_laplacian(g, normalized=False)
        L = lap.to_dense()
        row_sums = L.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_positive_semidefinite(self):
        """Unnormalized Laplacian eigenvalues >= 0."""
        g = self._make_chain(5)
        lap = build_laplacian(g, normalized=False)
        L = lap.to_dense()
        eigenvalues = np.linalg.eigvalsh(L)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_sparse_output(self):
        g = self._make_chain(4)
        lap = build_laplacian(g)
        assert sparse.issparse(lap.matrix)

    def test_random_walk_normalized(self):
        g = self._make_chain(4)
        lap = build_laplacian(g, normalized=True, laplacian_type="random_walk_normalized")
        L = lap.to_dense()
        # Diagonal should be 1.0
        assert L[0, 0] == pytest.approx(1.0)

    def test_chord_graph_laplacian(self):
        g = TensionGraph()
        for src, tgt in [("C", "G"), ("G", "Am"), ("Am", "F"), ("F", "Dm")]:
            g.add_edge(src, tgt, 1.0)
        lap = build_laplacian(g)
        L = lap.to_dense()
        assert L.shape == (5, 5)
        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(L)
        assert all(ev >= -1e-10 for ev in eigenvalues)
