# conservation-spectral-python

[![PyPI version](https://img.shields.io/pypi/v/cocapn)](https://pypi.org/project/cocapn/) [![SuperInstance](https://img.shields.io/badge/SuperInstance-Ecosystem-blue)](https://github.com/SuperInstance)



Python SDK for spectral graph conservation analysis — build tension graphs, compute Laplacian eigenvalues, track conservation ratios, detect anomalies, and verify spectral fingerprints.

## What This Gives You

- **Tension graphs** — weighted directed graphs with edge tension attributes
- **Laplacian decomposition** — eigenvalue/eigenvector computation via QR iteration
- **Conservation ratios** — CR = λ₂/λₙ with spectral gap and Cheeger constant
- **Conservation tracking** — time-series tracker with alerts on anomalous drift
- **Anomaly detection** — classify anomalies with severity and suggested corrections
- **Spectral fingerprints** — hash-based graph identity for cross-system comparison

## Quick Start

```python
from conservation_spectral import (
    TensionGraph, build_laplacian, eigendecompose,
    conservation_ratio, analyze, ConservationTracker, detect_anomalies
)

# Build a tension graph
g = TensionGraph()
g.add_edge("A", "B", tension=0.8)
g.add_edge("B", "C", tension=0.5)
g.add_edge("C", "A", tension=0.3)

# Spectral analysis
L = build_laplacian(g)
eigen = eigendecompose(L)
cr = conservation_ratio(eigen.eigenvalues)
print(f"Conservation ratio: {cr:.4f}")

# Full analysis report
report = analyze(g)
print(report)
# ConservationReport(spectral_gap=..., cheeger=..., anomalous=False, ...)

# Track over time
tracker = ConservationTracker(window=100)
for observation in time_series:
    alert = tracker.observe(observation)
    if alert:
        print(f"⚠ {alert}")

# Anomaly detection
anomalies = detect_anomalies(g)
for a in anomalies:
    print(f"{a.severity}: {a.description}")
    print(f"  Fix: {a.suggestion}")
```

## API Reference

| Module | Key Functions | Description |
|---|---|---|
| `graph` | `TensionGraph` | Weighted directed graph with tension |
| `laplacian` | `build_laplacian` | Graph → Laplacian matrix |
| `eigen` | `eigendecompose` | Eigenvalue/eigenvector computation |
| `conservation` | `conservation_ratio`, `spectral_gap`, `cheeger_constant` | Core metrics |
| `tracker` | `ConservationTracker`, `Alert` | Time-series monitoring |
| `fingerprint` | `spectral_fingerprint`, `compare_fingerprints` | Graph identity |
| `anomaly` | `detect_anomalies`, `Anomaly`, `Fix` | Anomaly detection and repair |

## How It Fits

The **Python SDK** of the conservation spectral ecosystem:

- [conservation-spectral-js](https://github.com/SuperInstance/conservation-spectral-js) — TypeScript SDK (same API)
- [conservation-spectral-ada](https://github.com/SuperInstance/conservation-spectral-ada) — Ada port (DO-178C certified)
- [conservation-protocol](https://github.com/SuperInstance/conservation-protocol) — Rust messaging protocol
- [conservation-conformance](https://github.com/SuperInstance/conservation-conformance) — cross-language conformance tests
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — uses conservation for constraint verification

## Testing

```bash
pip install -e ".[dev]"
pytest -v  # 5 test files
```

## Installation

```bash
pip install conservation-spectral
```

Requires Python ≥ 3.10.

## License

MIT

## Documentation

📚 [OpenConstruct Docs](https://github.com/SuperInstance/openconstruct-docs)
