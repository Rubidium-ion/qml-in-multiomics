#!/usr/bin/env python3
"""sweep_vqc.py — small grid search over vqc_qiskit encoding/ansatz params.

authors 2026-07-31. Purpose: find whether the VQC underperformance is an
encoding problem, a capacity problem, or an optimiser problem.
"""
from __future__ import annotations
import itertools, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from algorithms.vqc_qiskit import Algorithm  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

D = os.path.join(REPO, "data", "tcga_brca_v1")
Xtr, Xte = np.load(f"{D}/X_train.npy"), np.load(f"{D}/X_test.npy")
ytr, yte = np.load(f"{D}/y_train.npy"), np.load(f"{D}/y_test.npy")

GRID = list(itertools.product(
    [4, 8],            # n_qubits
    [1, 2],            # reps_ansatz
    ["cobyla"],        # optimizer
    [np.pi / 2, np.pi],  # angle_scale
))

rows = []
for nq, ra, opt, ang in GRID:
    cfg = {"params": {"n_qubits": nq, "reps_fm": 1, "reps_ansatz": ra,
                      "maxiter": 300, "optimizer": opt,
                      "angle_scale": float(ang), "seed": 42}}
    t0 = time.perf_counter()
    a = Algorithm(cfg).train(Xtr, ytr)
    yp = a.predict(Xte)
    pr = a.predict_proba(Xte)
    row = {
        "n_qubits": nq, "reps_ansatz": ra, "optimizer": opt,
        "angle_scale": round(float(ang), 4),
        "accuracy": float(accuracy_score(yte, yp)),
        "f1": float(f1_score(yte, yp, zero_division=0)),
        "auc_roc": float(roc_auc_score(yte, pr[:, 1])),
        "train_time": round(time.perf_counter() - t0, 1),
        "fallback": a._used_fallback,
    }
    rows.append(row)
    print(f"nq={nq} ra={ra} ang={ang:.2f} -> acc={row['accuracy']:.3f} "
          f"f1={row['f1']:.3f} auc={row['auc_roc']:.3f} ({row['train_time']}s)")

out = os.path.join(REPO, "benchmarks", "results", "vqc_sweep_001.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"grid": rows,
           "majority_baseline": float(max(np.bincount(yte)) / len(yte))},
          open(out, "w"), indent=2)
print("\nwrote", out)
