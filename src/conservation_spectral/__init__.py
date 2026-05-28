"""Conservation Spectral SDK — spectral analysis of tension graphs."""

__version__ = "0.1.0"

from .graph import TensionGraph
from .laplacian import Laplacian, build_laplacian
from .eigen import EigenDecomposition, eigendecompose
from .conservation import (
    conservation_ratio,
    conservation_ratios,
    spectral_gap,
    cheeger_constant,
    analyze,
    ConservationReport,
    ConservationRatio,
)
from .tracker import ConservationTracker, Alert
from .fingerprint import spectral_fingerprint_hash, compare_fingerprints, spectral_fingerprint
from .anomaly import Anomaly, AnomalyType, Fix, detect_anomalies, suggest_correction

__all__ = [
    "TensionGraph",
    "Laplacian",
    "build_laplacian",
    "EigenDecomposition",
    "eigendecompose",
    "conservation_ratio",
    "conservation_ratios",
    "spectral_gap",
    "cheeger_constant",
    "analyze",
    "ConservationReport",
    "ConservationRatio",
    "ConservationTracker",
    "Alert",
    "spectral_fingerprint_hash",
    "compare_fingerprints",
    "spectral_fingerprint",
    "Anomaly",
    "AnomalyType",
    "Fix",
    "detect_anomalies",
    "suggest_correction",
]
