"""Spectral fingerprinting — hashing and comparison of eigenvalue spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .eigen import EigenDecomposition


@dataclass
class SpectralFingerprint:
    """Summary statistics of the eigenspectrum."""
    eigenvalue_histogram: np.ndarray
    spectral_entropy: float
    effective_dimension: float
    gap_profile: np.ndarray
    conservation_profile: np.ndarray  # filled from conservation ratios


def spectral_fingerprint(
    eigen: EigenDecomposition,
    ratios: Optional[list] = None,
) -> SpectralFingerprint:
    """Compute spectral fingerprint from eigendecomposition.

    Args:
        eigen: EigenDecomposition result.
        ratios: Optional conservation ratios to include in profile.

    Returns:
        SpectralFingerprint with summary statistics.
    """
    evals = eigen.eigenvalues

    # Eigenvalue histogram (binned into sqrt(n) bins)
    n_bins = max(2, int(np.ceil(np.sqrt(len(evals)))))
    if len(evals) > 1:
        emin, emax = float(evals.min()), float(evals.max())
        if emax - emin < 1e-15:
            histogram = np.zeros(n_bins)
            histogram[0] = len(evals)
        else:
            histogram, _ = np.histogram(evals, bins=n_bins)
            histogram = histogram.astype(np.float64)
    else:
        histogram = np.array([1.0])

    # Spectral entropy: H = -Σ p_i log(p_i) where p_i = λ_i / Σ λ_i
    total = float(np.sum(np.abs(evals)))
    if total > 1e-15:
        probs = np.abs(evals) / total
        probs = probs[probs > 1e-15]
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    # Effective dimension: exp(entropy) (perplexity)
    effective_dim = float(np.exp(entropy))

    # Gap profile: consecutive eigenvalue differences
    if len(evals) > 1:
        gaps = np.diff(evals)
    else:
        gaps = np.array([])

    # Conservation profile
    if ratios:
        conservation_profile = np.array([r.ratio if hasattr(r, 'ratio') else r for r in ratios])
    else:
        conservation_profile = np.array([])

    return SpectralFingerprint(
        eigenvalue_histogram=histogram,
        spectral_entropy=entropy,
        effective_dimension=effective_dim,
        gap_profile=gaps,
        conservation_profile=conservation_profile,
    )


def spectral_fingerprint_hash(eigenvalues: np.ndarray, precision: int = 6) -> str:
    """Compute a BLAKE3-like hash of the rounded eigenvalue spectrum.

    Uses hashlib if blake3 is not available.

    Args:
        eigenvalues: Array of eigenvalues.
        precision: Decimal places to round to before hashing.

    Returns:
        Hex string of the hash.
    """
    rounded = np.round(eigenvalues, precision).tobytes()

    try:
        import blake3
        return blake3.blake3(rounded).hexdigest()
    except ImportError:
        import hashlib
        return hashlib.sha256(rounded).hexdigest()


def compare_fingerprints(fp1: SpectralFingerprint, fp2: SpectralFingerprint) -> float:
    """Compare two spectral fingerprints and return a similarity score [0, 1].

    Uses a combination of histogram similarity, entropy similarity, and gap profile correlation.

    Returns:
        Float between 0 (completely different) and 1 (identical).
    """
    # Histogram similarity (cosine similarity)
    h1, h2 = fp1.eigenvalue_histogram, fp2.eigenvalue_histogram
    if len(h1) == 0 or len(h2) == 0:
        hist_sim = 0.0
    else:
        # Pad to same length
        max_len = max(len(h1), len(h2))
        h1p = np.pad(h1, (0, max_len - len(h1)))
        h2p = np.pad(h2, (0, max_len - len(h2)))
        dot = np.dot(h1p, h2p)
        norm1 = np.linalg.norm(h1p)
        norm2 = np.linalg.norm(h2p)
        if norm1 > 1e-15 and norm2 > 1e-15:
            hist_sim = float(dot / (norm1 * norm2))
        else:
            hist_sim = 0.0

    # Entropy similarity
    ent1, ent2 = fp1.spectral_entropy, fp2.spectral_entropy
    if max(ent1, ent2) > 1e-15:
        ent_sim = 1.0 - abs(ent1 - ent2) / max(ent1, ent2)
    else:
        ent_sim = 1.0

    # Gap profile similarity (Pearson correlation)
    g1, g2 = fp1.gap_profile, fp2.gap_profile
    if len(g1) > 1 and len(g2) > 1:
        min_len = min(len(g1), len(g2))
        g1r, g2r = g1[:min_len], g2[:min_len]
        if np.std(g1r) > 1e-15 and np.std(g2r) > 1e-15:
            gap_sim = float(np.corrcoef(g1r, g2r)[0, 1])
            gap_sim = max(0.0, gap_sim)  # clamp negative
        else:
            gap_sim = 1.0 if np.allclose(g1r, g2r, atol=1e-10) else 0.0
    else:
        gap_sim = 1.0

    # Weighted average
    return float(0.4 * hist_sim + 0.3 * ent_sim + 0.3 * gap_sim)


def identify_language(text: str) -> str:
    """Identify the language of text using token transition fingerprinting.

    This is a simplified demonstration. Builds a transition graph from
    character transitions and uses spectral properties to classify.

    Args:
        text: Input text.

    Returns:
        Detected language name.
    """
    # Build character transition frequencies
    char_set = sorted(set(text.lower()))
    if len(char_set) < 3:
        return "unknown"

    char_idx = {c: i for i, c in enumerate(char_set)}
    n = len(char_set)
    transitions = np.zeros((n, n), dtype=np.float64)

    for i in range(len(text) - 1):
        c1 = text[i].lower()
        c2 = text[i + 1].lower()
        if c1 in char_idx and c2 in char_idx:
            transitions[char_idx[c1], char_idx[c2]] += 1.0

    # Row-normalize
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transitions /= row_sums

    # Build graph and compute spectrum
    from .graph import TensionGraph
    from .laplacian import build_laplacian
    from .eigen import eigendecompose

    graph = TensionGraph(directed=True)
    for i, c in enumerate(char_set):
        graph.add_vertex(c)
    for i in range(n):
        for j in range(n):
            if transitions[i, j] > 0.01:
                graph.add_edge(char_set[i], char_set[j], transitions[i, j])

    if graph.vertex_count < 2:
        return "unknown"

    lap = build_laplacian(graph)
    eigen = eigendecompose(lap, num_vectors=min(5, graph.vertex_count - 1))

    # Use spectral entropy and effective dimension as features
    fp = spectral_fingerprint(eigen)

    # Very rough heuristic language classification
    # (A real system would use trained classifiers)
    ed = fp.effective_dimension
    se = fp.spectral_entropy

    if ed < 3:
        return "japanese" if len(char_set) > 30 else "english"
    elif ed < 6:
        return "french"
    elif ed < 10:
        return "german"
    else:
        return "unknown"
