#!/usr/bin/env python
"""test_multiregister.py — guard tests for vqc_multiregister.

Covers the three things that actually broke before:
  T1 register dims are parameterised end-to-end (authors' request)
  T2 NO PCA leakage: transforms fitted on train only
  T3 predict_proba returns real probabilities, not one-hot hard labels
  T4 cross-register information actually flows (CRZ regression guard)

Run:  .venv/bin/python scripts/test_multiregister.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.vqc_multiregister.model import Algorithm  # noqa: E402
from algorithms.vqc_multiregister.circuit import (  # noqa: E402
    MultiRegisterConfig, RegisterSpec, circuit_stats,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_paired(n=60, n_shape=14, n_tex=88, n_omics=500, seed=0):
    """Synthetic paired radiomics+omics with signal only in texture+omics."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = rng.normal(0, 1, (n, n_shape))
    tex = rng.normal(0, 1, (n, n_tex)) + 0.9 * y[:, None] * (np.arange(n_tex) < 5)
    omics = rng.normal(0, 50, (n, n_omics)) + 30 * y[:, None] * (np.arange(n_omics) < 10)
    X = np.hstack([shape, tex, omics])
    blocks = {
        "shape": [0, n_shape],
        "texture": [n_shape, n_shape + n_tex],
        "transcriptomics": [n_shape + n_tex, n_shape + n_tex + n_omics],
    }
    return X, y, blocks


def cfg_for(blocks, dims):
    return {"params": {
        "encoder": "ry", "coupling": "ring", "coupler": "cry",
        "registers": [
            {"name": "shape", "n_qubits": dims[0]},
            {"name": "texture", "n_qubits": dims[1]},
            {"name": "transcriptomics", "n_qubits": dims[2], "angle_scale": np.pi / 4},
        ],
        "blocks": blocks, "maxiter": 8, "reps_ansatz": 1, "seed": 42,
    }}


# --------------------------------------------------------------------------- #
print("\nT1 · register dims parameterised")
X, y, blocks = make_paired()
for dims in [(2, 2, 2), (3, 4, 4), (2, 3, 5), (1, 1, 1)]:
    algo = Algorithm(cfg_for(blocks, dims))
    Xt = algo._fit_transform_blocks(X)
    check(f"dims {dims} -> width {Xt.shape[1]}", Xt.shape[1] == sum(dims))
    st = circuit_stats(algo._make_circuit_config())
    check(f"dims {dims} -> circuit {st['n_qubits']}q", st["n_qubits"] == sum(dims))

algo = Algorithm(cfg_for(blocks, (3, 4, 4)))
Xt = algo._fit_transform_blocks(X)
sc = [r["angle_scale"] for r in algo.register_specs]
omics_max = np.abs(Xt[:, 7:11]).max()
check("per-register angle_scale honoured", omics_max <= sc[2] + 1e-9,
      f"omics |max|={omics_max:.4f} <= pi/4={sc[2]:.4f}")


# --------------------------------------------------------------------------- #
print("\nT2 · no PCA leakage (the 2026-07-31 P0 bug)")
n_tr = 40
Xtr, Xte = X[:n_tr], X[n_tr:]
algo = Algorithm(cfg_for(blocks, (3, 4, 4)))
Ztr = algo._fit_transform_blocks(Xtr)
Zte = algo._transform_blocks(Xte)

# a leaked PCA centres the FULL set; a clean one centres only train
pca = algo._pipelines["transcriptomics"].named_steps.get("pca")
Xb_tr = Xtr[:, blocks["transcriptomics"][0]:blocks["transcriptomics"][1]]
Xb_all = X[:, blocks["transcriptomics"][0]:blocks["transcriptomics"][1]]
std = algo._pipelines["transcriptomics"].named_steps["std"]
mean_tr = np.abs(std.transform(Xb_tr).mean(axis=0)).max()
mean_all = np.abs(std.transform(Xb_all).mean(axis=0)).max()
check("scaler centres TRAIN (~0)", mean_tr < 1e-8, f"{mean_tr:.2e}")
check("scaler does NOT centre full set", mean_all > 1e-3, f"{mean_all:.2e}")

# transform must be pure: repeated calls identical, and test rows must not
# change when other test rows are removed
z_single = algo._transform_blocks(Xte[:1])
check("transform is row-independent", np.allclose(z_single, Zte[:1], atol=1e-12))
check("transform is deterministic", np.allclose(algo._transform_blocks(Xte), Zte))

# refitting on a different split must give a different projection
algo2 = Algorithm(cfg_for(blocks, (3, 4, 4)))
algo2._fit_transform_blocks(X[20:])
pca2 = algo2._pipelines["transcriptomics"].named_steps.get("pca")
diff = np.abs(pca.components_ - pca2.components_).max()
check("PCA basis depends on training fold", diff > 1e-6, f"max delta {diff:.4f}")


# --------------------------------------------------------------------------- #
print("\nT3 · predict_proba returns probabilities")
algo = Algorithm(cfg_for(blocks, (2, 2, 2)))
algo.train(Xtr, y[:n_tr])
p = algo.predict_proba(Xte)
pred = algo.predict(Xte)
check("proba shape", p.shape == (len(Xte), 2), str(p.shape))
check("rows sum to 1", np.allclose(p.sum(axis=1), 1.0, atol=1e-6))
check("not one-hot hard labels", not np.all(np.isin(p, [0.0, 1.0])),
      f"unique~{len(np.unique(np.round(p,4)))}")
check("in [0,1]", p.min() >= -1e-9 and p.max() <= 1 + 1e-9)
check("predict shape", pred.shape == (len(Xte),))
check("quantum path used (not fallback)", not algo._used_fallback)


# --------------------------------------------------------------------------- #
print("\nT4 · cross-register info flow (CRZ regression guard)")
from scripts.encoding_probe import d3_cross_register  # noqa: E402

rng = np.random.default_rng(7)
c_cry = MultiRegisterConfig(coupling="ring", coupler="cry", encoder="ry")
c_crz = MultiRegisterConfig(coupling="ring", coupler="crz", encoder="ry")
r_cry = d3_cross_register(c_cry, np.random.default_rng(7))
r_crz = d3_cross_register(c_crz, np.random.default_rng(7))
check("cry transports information", r_cry["information_flows"],
      f"shift {r_cry['shape_marginal_shift']:.5f}")
check("crz stays diagonal (known-dead)", not r_crz["information_flows"],
      f"shift {r_crz['shape_marginal_shift']:.5f}")
check("default coupler is cry", MultiRegisterConfig().coupler == "cry")


# --------------------------------------------------------------------------- #
print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all checks passed")
