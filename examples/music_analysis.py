"""Music analysis example — Bach chorale-style chord progression."""

import numpy as np
from conservation_spectral.graph import TensionGraph
from conservation_spectral.laplacian import build_laplacian
from conservation_spectral.eigen import eigendecompose
from conservation_spectral.conservation import analyze, conservation_ratios
from conservation_spectral.fingerprint import spectral_fingerprint, spectral_fingerprint_hash
from conservation_spectral.anomaly import detect_anomalies


def main():
    print("🎵 Conservation Spectral SDK — Music Analysis Demo\n")
    print("=" * 60)

    # Build a Bach-style chorale progression
    # Typical functional harmony: I → IV → V → I → vi → ii → V → I
    # In C major: C → F → G → C → Am → Dm → G → C
    progression = [
        ("C", "F"),    # I → IV
        ("F", "G"),    # IV → V
        ("G", "C"),    # V → I (perfect cadence)
        ("C", "Am"),   # I → vi (deceptive)
        ("Am", "Dm"),  # vi → ii
        ("Dm", "G"),   # ii → V (ii-V-I)
        ("G", "C"),    # V → I
    ]

    # Build the tension graph from chord transitions
    graph = TensionGraph.build_from_transitions(progression)

    # Add tension values based on music theory
    # Tension heuristic: I=0.1, IV=0.3, V=0.7, vi=0.4, ii=0.5
    tension_map = {"C": 0.1, "F": 0.3, "G": 0.7, "Am": 0.4, "Dm": 0.5}
    tension = np.array([tension_map[v] for v in graph.vertices], dtype=np.float64)
    graph.set_attribute("tension", tension)

    print(f"\n📊 Graph: {graph.vertex_count} vertices, {graph.edge_count} edges")
    print(f"   Vertices: {graph.vertices}")
    print(f"   Tension:  {tension.tolist()}")

    # Full conservation analysis
    print("\n🔬 Conservation Analysis:")
    print("-" * 40)
    report = analyze(graph, "tension", tension)

    print(f"  Spectral gap:     {report.spectral_gap:.4f}")
    print(f"  Cheeger constant: {report.cheeger_constant:.4f}")

    print("\n  Conservation Ratios (per eigenvector):")
    for r in report.ratios:
        print(f"    EV[{r.eigenvector_index}] λ={r.eigenvalue:.4f}  ratio={r.ratio:.6f}")

    # Spectral fingerprint
    print("\n🧬 Spectral Fingerprint:")
    print("-" * 40)
    fp = report.fingerprint
    print(f"  Spectral entropy:     {fp.spectral_entropy:.4f}")
    print(f"  Effective dimension:  {fp.effective_dimension:.4f}")
    print(f"  Gap profile:          {[f'{g:.4f}' for g in fp.gap_profile]}")

    # Fingerprint hash
    lap = build_laplacian(graph)
    eigen = eigendecompose(lap)
    fp_hash = spectral_fingerprint_hash(eigen.eigenvalues)
    print(f"  Fingerprint hash:     {fp_hash[:32]}...")

    # Anomaly detection
    print("\n🔍 Anomaly Detection:")
    print("-" * 40)
    anomalies = detect_anomalies(graph, threshold=1.5)
    if anomalies:
        for a in anomalies:
            print(f"  ⚠️  {a.description}")
    else:
        print("  ✅ No anomalies detected (threshold=1.5σ)")

    # Compare with a different progression
    print("\n📈 Comparison: Bach vs Pop Progression:")
    print("-" * 40)

    # Pop: I-V-vi-IV (C-G-Am-F) repeated
    pop_progression = [
        ("C", "G"), ("G", "Am"), ("Am", "F"), ("F", "C"),
        ("C", "G"), ("G", "Am"), ("Am", "F"), ("F", "C"),
    ]
    pop_graph = TensionGraph.build_from_transitions(pop_progression)
    pop_tension = np.array([tension_map.get(v, 0.3) for v in pop_graph.vertices], dtype=np.float64)
    pop_graph.set_attribute("tension", pop_tension)

    pop_report = analyze(pop_graph, "tension", pop_tension)

    print(f"  Bach — spectral gap: {report.spectral_gap:.4f}, entropy: {fp.spectral_entropy:.4f}")
    print(f"  Pop  — spectral gap: {pop_report.spectral_gap:.4f}, entropy: {pop_report.fingerprint.spectral_entropy:.4f}")

    from conservation_spectral.fingerprint import compare_fingerprints
    similarity = compare_fingerprints(fp, pop_report.fingerprint)
    print(f"  Similarity: {similarity:.4f}")

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
