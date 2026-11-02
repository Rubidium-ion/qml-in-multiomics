#!/usr/bin/env python
"""encoding_probe.py - answer "Hadamard init vs Angle Encoding?" with numbers.

Runs three diagnostics on the multi-register feature map, on synthetic data
shaped like authors' packing ([shape, texture, transcriptomics]) so it can run
before TCIA BRCA-Radiomics finishes downloading.

  D1 Z-basis degeneracy - do different inputs give different measurement
     statistics? (H+RZ without an entangler: NO.)
  D2 Kernel expressivity - spread of the fidelity kernel off-diagonal.
     Too flat -> can't separate. Too close to identity -> memorises.
  D3 Cross-register information - does the coupling actually let a
     transcriptomics feature change a shape-register marginal?

Usage:
    python scripts/encoding_probe.py --out benchmarks/results/encoding_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.vqc_multiregister.circuit import (  # noqa: E402
    MultiRegisterConfig,
    RegisterSpec,
    build_feature_map,
    circuit_stats,
)

ENCODERS = ["ry", "h_rz", "zz", "ry_zz"]


def make_config(encoder: str, coupling: str, coupler: str = "cry") -> MultiRegisterConfig:
    return MultiRegisterConfig(
        registers=[
            RegisterSpec("shape", 3),
            RegisterSpec("texture", 4),
            RegisterSpec("transcriptomics", 4, angle_scale=np.pi / 4),
        ],
        encoder=encoder,
        coupling=coupling,
        coupler=coupler,
    )


def statevector(cfg, x):
    from qiskit.quantum_info import Statevector

    qc, params = build_feature_map(cfg)
    return Statevector.from_instruction(qc.assign_parameters({params: list(x)}))


def z_probabilities(cfg, x) -> np.ndarray:
    """Per-qubit P(|1>) - what a Z-basis measurement actually sees."""
    sv = statevector(cfg, x)
    n = cfg.n_qubits
    probs = np.abs(np.asarray(sv.data)) ** 2
    out = np.zeros(n)
    for basis_idx, p in enumerate(probs):
        if p < 1e-15:
            continue
        for q in range(n):
            if (basis_idx >> q) & 1:
                out[q] += p
    return out


def d1_degeneracy(cfg, X) -> dict:
    P = np.stack([z_probabilities(cfg, x) for x in X])
    return {
        "marginal_std_mean": float(P.std(axis=0).mean()),
        "marginal_range_max": float((P.max(axis=0) - P.min(axis=0)).max()),
        "degenerate": bool((P.max(axis=0) - P.min(axis=0)).max() < 1e-9),
    }


def d2_kernel(cfg, X) -> dict:
    S = np.stack([np.asarray(statevector(cfg, x).data) for x in X])
    K = np.abs(S.conj() @ S.T) ** 2
    off = K[~np.eye(len(X), dtype=bool)]
    return {
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "offdiag_min": float(off.min()),
        "offdiag_max": float(off.max()),
        # concentration -> exponentially flat kernel -> untrainable
        "concentrated": bool(off.std() < 1e-3),
    }


def d3_cross_register(cfg, rng) -> dict:
    """Perturb ONE transcriptomics feature; measure response in shape register."""
    n = cfg.n_qubits
    spans = cfg.slices()
    x0 = rng.uniform(-1, 1, n)
    x1 = x0.copy()
    x1[spans["transcriptomics"].start] += 0.8

    p0 = z_probabilities(cfg, x0)
    p1 = z_probabilities(cfg, x1)
    sh = spans["shape"]
    delta = float(np.abs(p1[sh] - p0[sh]).max())
    return {
        "shape_marginal_shift": delta,
        "information_flows": bool(delta > 1e-6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmarks/results/encoding_probe.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_features = make_config("ry", "none").n_qubits
    X = rng.uniform(-1.0, 1.0, size=(args.n_samples, n_features))

    results = []
    for coupling, coupler in (("none", "cry"), ("ring", "crz"), ("ring", "cry")):
        for enc in ENCODERS:
            cfg = make_config(enc, coupling, coupler)
            row = {
                "encoder": enc,
                "coupling": coupling,
                "coupler": coupler,
                "stats": circuit_stats(cfg),
                "d1_z_degeneracy": d1_degeneracy(cfg, X),
                "d2_kernel": d2_kernel(cfg, X),
                "d3_cross_register": d3_cross_register(cfg, rng),
            }
            results.append(row)
            d1, d2, d3 = row["d1_z_degeneracy"], row["d2_kernel"], row["d3_cross_register"]
            print(
                f"{enc:>6} / {coupling}-{coupler:<4} | depth {row['stats']['depth']:>3} "
                f"2q {row['stats']['two_qubit_gates']:>3} | "
                f"Zspread {d1['marginal_range_max']:.4f} "
                f"{'DEGENERATE' if d1['degenerate'] else '          '} | "
                f"K_off {d2['offdiag_mean']:.4f}+-{d2['offdiag_std']:.4f} | "
                f"xreg {d3['shape_marginal_shift']:.5f}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "author": "authors",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "note": "synthetic uniform features; structural probe only, no labels",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
