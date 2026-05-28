"""Shared types to avoid circular imports."""
from dataclasses import dataclass
from typing import Any

@dataclass
class ConservationRatio:
    mode: int
    eigenvalue: float
    ratio: float

@dataclass  
class SpectralFingerprintData:
    eigenvalue_histogram: Any  # np.ndarray
    entropy: float
    effective_dimension: int
    gap_profile: Any  # np.ndarray
