"""Tests for TensionGraph."""

import numpy as np
import pytest

from conservation_spectral.graph import TensionGraph


class TestTensionGraph:
    def test_empty_graph(self):
        g = TensionGraph()
        assert g.vertex_count == 0
        assert g.edge_count == 0

    def test_add_vertices(self):
        g = TensionGraph()
        idx_c = g.add_vertex("C")
        idx_g = g.add_vertex("G")
        idx_am = g.add_vertex("Am")
        assert g.vertex_count == 3
        assert idx_c == 0
        assert idx_g == 1

    def test_add_edges(self):
        g = TensionGraph()
        g.add_edge("C", "G", 1.0)
        g.add_edge("G", "Am", 0.8)
        assert g.vertex_count == 3
        assert g.edge_count == 2

    def test_adjacency_matrix(self):
        g = TensionGraph()
        g.add_edge("C", "G", 1.0)
        g.add_edge("G", "Am", 0.8)
        W = g.adjacency_matrix()
        assert W.shape[0] == W.shape[1]  # square matrix
        assert W.shape[0] >= 2  # at least 2 vertices connected
        assert W[0, 1] == 1.0  # C → G
        assert W[1, 2] == 0.8  # G → Am

    def test_adjacency_sparse(self):
        g = TensionGraph()
        g.add_edge("C", "G", 1.0)
        W = g.adjacency_sparse()
        assert W.shape[0] == W.shape[1]  # square matrix
        assert W.shape[0] >= 2  # at least 2 vertices connected
        assert W[0, 1] == 1.0

    def test_chord_progression(self):
        """Musical chord progression: C → G → Am → F → Dm"""
        g = TensionGraph()
        chords = ["C", "G", "Am", "F", "Dm"]
        transitions = [("C", "G"), ("G", "Am"), ("Am", "F"), ("F", "Dm")]
        for src, tgt in transitions:
            g.add_edge(src, tgt, 1.0)

        assert g.vertex_count == 5
        assert g.edge_count == 4

        W = g.adjacency_matrix()
        # C → G (idx 0→1), G → Am (idx 1→2), Am → F (idx 2→3), F → Dm (idx 3→4)
        assert W[0, 1] == 1.0
        assert W[1, 2] == 1.0
        assert W[2, 3] == 1.0
        assert W[3, 4] == 1.0

    def test_build_from_transitions(self):
        transitions = [
            ("C", "G"), ("G", "Am"), ("Am", "F"),
            ("C", "G"), ("G", "C"),
        ]
        g = TensionGraph.build_from_transitions(transitions)
        assert g.vertex_count == 4
        W = g.adjacency_matrix()
        # C → G happened twice
        ci = g._vertex_index["C"]
        gi = g._vertex_index["G"]
        assert W[ci, gi] == 2.0

    def test_attributes(self):
        g = TensionGraph()
        for v in ["A", "B", "C"]:
            g.add_vertex(v)
        g.set_attribute("tension", np.array([0.1, 0.5, 0.9]))
        attr = g.get_attribute("tension")
        np.testing.assert_array_almost_equal(attr, [0.1, 0.5, 0.9])

    def test_duplicate_vertex(self):
        g = TensionGraph()
        i1 = g.add_vertex("X")
        i2 = g.add_vertex("X")
        assert i1 == i2
        assert g.vertex_count == 1

    def test_undirected_graph(self):
        g = TensionGraph(directed=False)
        g.add_edge("A", "B", 2.0)
        W = g.adjacency_matrix()
        assert W[0, 1] == 2.0
        assert W[1, 0] == 2.0
