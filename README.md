# conservation-spectral-python

Python SDK for spectral graph theory conservation analysis — Laplacian construction, eigendecomposition, anomaly detection, and spectral fingerprinting.

## What This Gives You

- **Tension graphs** — Weighted undirected graphs with vertex attributes for spectral analysis
- **Laplacian matrices** — Unnormalized, normalized, and random-walk variants, built from graph topology
- **Eigendecomposition** — Full eigenvalue/eigenvector computation via SciPy
- **Conservation ratios** — Quantify how well each eigenvector preserves vertex attributes
- **Anomaly detection** — Identify conservation violations, structural breaks, and spectral outliers with fix suggestions
- **Spectral fingerprints** — BLAKE3-hashed eigenvalue signatures for comparing graph states
- **Real-time tracking** — Sliding-window conservation monitor with configurable alerts

## Quick Start

```python
from conservation_spectral import (
    TensionGraph, build_laplacian, eigendecompose,
    conservation_ratios, spectral_gap, cheeger_constant,
    analyze, detect_anomalies, spectral_fingerprint,
)

# Build a chord progression graph
g = TensionGraph(directed=False)
for chord in ["C", "G", "Am", "F", "Dm"]:
    g.add_vertex(chord)
g.add_edge("C", "G", 0.8)
g.add_edge("G", "Am", 0.6)
g.add_edge("Am", "F", 0.4)
g.add_edge("F", "Dm", 0.3)

# Spectral analysis
lap = build_laplacian(g)
eigen = eigendecompose(lap)
report = analyze(eigen, g)

print(f"Spectral gap: {report.spectral_gap:.4f}")
print(f"Cheeger constant: {report.cheeger_constant:.4f}")
```

See [`examples/music_analysis.py`](examples/music_analysis.py) for a full Bach chorale analysis demo.

## API Reference

| Module | Key Functions |
|--------|--------------|
| `graph` | `TensionGraph`, `add_vertex`, `add_edge` |
| `laplacian` | `build_laplacian`, `Laplacian` |
| `eigen` | `eigendecompose`, `EigenDecomposition` |
| `conservation` | `conservation_ratio`, `spectral_gap`, `cheeger_constant`, `analyze` |
| `tracker` | `ConservationTracker`, `Alert` |
| `fingerprint` | `spectral_fingerprint`, `spectral_fingerprint_hash`, `compare_fingerprints` |
| `anomaly` | `detect_anomalies`, `Anomaly`, `AnomalyType`, `Fix` |

## How It Fits

Part of the conservation spectral ecosystem — this is the **Python implementation**. Cross-language siblings:

- **Rust**: [conservation-spectral](https://github.com/SuperInstance/conservation-spectral) — core engine
- **TypeScript**: [conservation-spectral-js](https://github.com/SuperInstance/conservation-spectral-js) — JS/TS SDK
- **Ada**: [conservation-spectral-ada](https://github.com/SuperInstance/conservation-spectral-ada) — DO-178C certified
- **Conformance**: [conservation-conformance](https://github.com/SuperInstance/conservation-conformance) — cross-language test suite

## Testing

```bash
pip install -e ".[dev]"
pytest
```

5 test modules covering graph construction, Laplacian building, eigendecomposition, conservation analysis, and real-time tracking.

## Installation

```bash
pip install conservation-spectral
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10. Optional: `blake3` for fingerprint hashing.

## License

MIT
