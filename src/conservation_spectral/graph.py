"""TensionGraph — weighted graph with vertex attributes and transition probabilities."""

from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
from scipy import sparse


class TensionGraph:
    """Weighted directed graph with named vertex attributes.

    Stores vertices, edges, transition probabilities, and per-vertex attributes.
    """

    def __init__(self, directed: bool = True) -> None:
        self.directed = directed
        self._vertices: list[Any] = []
        self._vertex_index: dict[Any, int] = {}
        self._edges: list[tuple[int, int, float]] = []
        self._adjacency: dict[int, list[tuple[int, float]]] = {}
        self._attributes: dict[str, np.ndarray] = {}

    def add_vertex(self, vertex: Any, attribute: Optional[dict[str, float]] = None) -> int:
        """Add a vertex. Returns its index."""
        if vertex in self._vertex_index:
            return self._vertex_index[vertex]
        idx = len(self._vertices)
        self._vertices.append(vertex)
        self._vertex_index[vertex] = idx
        self._adjacency[idx] = []
        return idx

    def add_edge(self, source: Any, target: Any, weight: float = 1.0) -> None:
        """Add a weighted edge between two vertices."""
        si = self.add_vertex(source)
        ti = self.add_vertex(target)
        self._edges.append((si, ti, weight))
        self._adjacency[si].append((ti, weight))
        if not self.directed:
            self._adjacency[ti].append((si, weight))

    def set_attribute(self, name: str, values: np.ndarray) -> None:
        """Attach a named float array to vertices (indexed by vertex order)."""
        values = np.asarray(values, dtype=np.float64)
        if len(values) != self.vertex_count:
            raise ValueError(
                f"Attribute length {len(values)} != vertex count {self.vertex_count}"
            )
        self._attributes[name] = values

    def get_attribute(self, name: str) -> np.ndarray:
        return self._attributes[name]

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def vertices(self) -> list[Any]:
        return list(self._vertices)

    @property
    def attributes(self) -> dict[str, np.ndarray]:
        return dict(self._attributes)

    def adjacency_matrix(self) -> np.ndarray:
        """Return dense (n, n) adjacency/weight matrix."""
        n = self.vertex_count
        W = np.zeros((n, n), dtype=np.float64)
        for si, ti, w in self._edges:
            W[si, ti] += w
            if not self.directed:
                W[ti, si] += w
        return W

    def adjacency_sparse(self) -> sparse.csr_matrix:
        """Return sparse CSR weight matrix."""
        return sparse.csr_matrix(self.adjacency_matrix())

    def degree_matrix(self) -> np.ndarray:
        """Diagonal degree matrix D where D[i,i] = sum of W[i,:]."""
        W = self.adjacency_matrix()
        return np.diag(W.sum(axis=1))

    @classmethod
    def build_from_transitions(
        cls,
        transitions: list[tuple[Any, Any]],
        similarity_fn: Optional[Callable[[Any, Any], float]] = None,
        directed: bool = True,
    ) -> TensionGraph:
        """Build a graph from a sequence of (from, to) transitions.

        Each transition increments the edge weight by 1 * similarity_fn(from, to).
        If similarity_fn is None, weight = count of transitions.
        """
        g = cls(directed=directed)
        counts: dict[tuple[int, int], float] = {}
        vertex_list: list[Any] = []

        for src, tgt in transitions:
            si = g.add_vertex(src)
            ti = g.add_vertex(tgt)
            sim = similarity_fn(src, tgt) if similarity_fn else 1.0
            key = (si, ti)
            counts[key] = counts.get(key, 0.0) + sim

        # Reset edges and rebuild with aggregated weights
        g._edges = []
        g._adjacency = {i: [] for i in range(g.vertex_count)}
        for (si, ti), w in counts.items():
            g._edges.append((si, ti, w))
            g._adjacency[si].append((ti, w))
            if not directed:
                g._adjacency[ti].append((si, w))

        return g
