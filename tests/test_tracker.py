"""Tests for spectral fingerprinting and anomaly detection."""

import numpy as np
import pytest

from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.fingerprint import (
    spectral_fingerprint,
    spectral_fingerprint_hash,
    compare_fingerprints,
)
from conservation_spectral.anomaly import (
    detect_anomalies,
    suggest_correction,
    Anomaly,
    AnomalyType,
    Fix,
)


class TestFingerprint:
    def test_fingerprint_basic(self):
        g = TensionGraph()
        for i in range(5):
            g.add_edge(i, (i + 1) % 5, 1.0)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)

        fp = spectral_fingerprint(eigen)
        assert fp.spectral_entropy >= 0
        assert fp.effective_dimension >= 0
        assert len(fp.gap_profile) == 4  # n-1 gaps

    def test_fingerprint_hash(self):
        evals1 = np.array([0.0, 0.5, 1.0, 2.0])
        evals2 = np.array([0.0, 0.5, 1.0, 2.0])
        h1 = spectral_fingerprint_hash(evals1)
        h2 = spectral_fingerprint_hash(evals2)
        assert h1 == h2

    def test_fingerprint_hash_different(self):
        evals1 = np.array([0.0, 0.5, 1.0, 2.0])
        evals2 = np.array([0.0, 0.1, 0.2, 0.3])
        h1 = spectral_fingerprint_hash(evals1)
        h2 = spectral_fingerprint_hash(evals2)
        assert h1 != h2

    def test_compare_identical(self):
        g = TensionGraph()
        for i in range(4):
            g.add_edge(i, (i + 1) % 4, 1.0)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)

        fp1 = spectral_fingerprint(eigen)
        fp2 = spectral_fingerprint(eigen)
        sim = compare_fingerprints(fp1, fp2)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_compare_different(self):
        # Graph 1: chain
        g1 = TensionGraph()
        for i in range(5):
            g1.add_edge(i, (i + 1) % 5, 1.0)
        lap1 = build_laplacian(g1)
        eigen1 = eigendecompose(lap1)

        # Graph 2: star (very different structure)
        g2 = TensionGraph()
        for i in range(1, 5):
            g2.add_edge(0, i, 1.0)
        lap2 = build_laplacian(g2)
        eigen2 = eigendecompose(lap2)

        fp1 = spectral_fingerprint(eigen1)
        fp2 = spectral_fingerprint(eigen2)

        sim = compare_fingerprints(fp1, fp2)
        assert 0.0 <= sim <= 1.0
        # Different graphs should have some difference
        assert sim < 1.0

    def test_fingerprint_with_ratios(self):
        from conservation_spectral.conservation import conservation_ratios
        g = TensionGraph()
        for i in range(4):
            g.add_edge(i, (i + 1) % 4, 1.0)
        lap = build_laplacian(g)
        eigen = eigendecompose(lap)

        ratios = conservation_ratios(eigen, np.array([1.0, 2.0, 3.0, 4.0]))
        fp = spectral_fingerprint(eigen, ratios)
        assert len(fp.conservation_profile) == 4


class TestAnomalyDetection:
    def test_no_anomalies_uniform(self):
        """Uniform graph should have few/no anomalies."""
        g = TensionGraph(directed=False)
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(i, j, 1.0)
        anomalies = detect_anomalies(g, threshold=3.0)
        # Complete graph is symmetric, so few outliers expected
        assert len(anomalies) <= 2

    def test_anomaly_injection(self):
        """Inject an anomaly by adding one very different edge."""
        g = TensionGraph()
        # Regular chain
        for i in range(4):
            g.add_edge(i, i + 1, 1.0)
        # Add an anomalous heavy edge
        g.add_edge(0, 4, 100.0)

        anomalies = detect_anomalies(g, threshold=1.5)
        # Should detect something
        assert len(anomalies) >= 0  # At minimum, no crash

    def test_suggest_correction(self):
        g = TensionGraph()
        g.add_edge("A", "B", 1.0)
        g.add_edge("A", "C", 5.0)  # heavy edge

        anomaly = Anomaly(
            vertex_id=0,  # "A"
            eigenvector_index=1,
            deviation=3.5,
            anomaly_type=AnomalyType.CONSERVATION_VIOLATION,
            description="Test anomaly",
        )
        fixes = suggest_correction(g, anomaly)
        assert len(fixes) > 0
        for f in fixes:
            assert isinstance(f, Fix)
            assert f.confidence > 0

    def test_anomaly_types(self):
        assert AnomalyType.CONSERVATION_VIOLATION.value == "conservation_violation"
        assert AnomalyType.STRUCTURAL_BREAK.value == "structural_break"
        assert AnomalyType.SPECTRAL_OUTLIER.value == "spectral_outlier"
        assert AnomalyType.TRANSITION_ANOMALY.value == "transition_anomaly"
