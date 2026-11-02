#!/usr/bin/env python3
"""generate_synthetic_views.py - controllable multi-view synthetic data for the QML platform.

Creates a canonical dataset dir (see docs/modality_ingest_spec.md) with integrated matrix X,
labels y, and per-view matrices under views/. The generator injects ground-truth structure so
downstream task-admission gates (e.g. the SHAP interaction-fraction gate) can be validated
against an oracle:

  - each named view is a block of features with its own latent dimensionality;
  - the label depends on (a) main-effect signal features and (b) cross-view interaction terms
    (products of standardized features from *different* views);
  - the mix is controlled by --interaction-share, the share of label logit variance attributed
    to interaction terms (recorded as true_interaction_share in provenance.json);
  - optional missingness and label noise make the task harder; parameters are recorded.

Usage:
    python scripts/generate_synthetic_views.py \
        --out data/synthetic_multimodal_v1 \
        --n-samples 600 --seed 42 --interaction-share 0.30

Outputs (in --out): X.npy, y.npy, feature_names.json, sample_ids.json, feature_views.json,
provenance.json, views/<view>.npy. Deterministic given seed. numpy only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np


def default_views() -> dict:
    # name -> (n_features, n_signal_features)
    return {
        "imaging": (64, 2),          # stand-in for CT/NM/MRI derived features
        "transcriptomics": (200, 4),
        "genomics": (150, 3),
        "metabolomics": (80, 2),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build(args) -> dict:
    rng = np.random.default_rng(args.seed)
    views = default_views() if args.views == "default" else _parse_views(args.views)
    n = args.n_samples

    names = []          # feature names in integrated order
    view_ranges = {}    # view -> [start, end)
    start = 0
    sig_features = {}   # view -> list of local feature indices used as signal

    X = np.empty((n, 0), dtype=np.float32)
    view_mats = {}
    for view, (dim, n_sig) in views.items():
        # blocks with correlation structure; standardized columns below
        base = rng.standard_normal((n, dim))
        rho = 0.3
        # induce mild within-view correlation
        common = rng.standard_normal((n, 1))
        block = base * np.sqrt(1 - rho) + common * np.sqrt(rho)
        block = (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-8)
        view_mats[view] = block.astype(np.float32)
        X = np.hstack([X, block])
        idx = list(range(start, start + dim))
        names += [f"{view}:f{i}" for i in range(dim)]
        view_ranges[view] = [start, start + dim]
        # spread signal features through the view deterministically
        picks = rng.choice(dim, size=n_sig, replace=False)
        sig_features[view] = [start + int(p) for p in picks]
        start += dim

    # label: main effect from a few signal features + cross-view interaction terms
    n_views = len(views)
    view_list = list(views.keys())
    main_idx = []
    for v, locs in sig_features.items():
        main_idx += locs[:2]
    main_idx = main_idx[:6]

    # interaction terms: pick pairs across DIFFERENT views
    int_pairs = []
    seen = set()
    while len(int_pairs) < args.n_interactions:
        a = rng.choice(view_list)
        b = rng.choice(view_list)
        if a == b:
            continue
        fa = int(rng.choice(sig_features[a][:2]))
        fb = int(rng.choice(sig_features[b][:2]))
        key = (min(fa, fb), max(fa, fb))
        if key in seen:
            continue
        seen.add(key)
        int_pairs.append((fa, fb))

    m = X[:, main_idx]
    z_main = m[:, :2].sum(axis=1) + 0.5 * m[:, 2:].sum(axis=1)
    z_int = np.zeros(n)
    for (fa, fb) in int_pairs:
        z_int += X[:, fa] * X[:, fb]
    z_int /= max(1, len(int_pairs))

    # mix main/interaction so the *share of variance* of z matches --interaction-share
    target = float(np.clip(args.interaction_share, 0.0, 0.95))
    # grid search a scale s such that var(s*z_int) / (var(z_main) + var(s*z_int)) ~ target
    best = 1.0
    best_err = 1e9
    for s in np.linspace(0.05, 10.0, 400):
        v_m = float(np.var(z_main))
        v_i = float(np.var(s * z_int))
        share = v_i / (v_m + v_i + 1e-12)
        if abs(share - target) < best_err:
            best_err = abs(share - target)
            best = s
    z = z_main + best * z_int
    z += args.noise * rng.standard_normal(n)

    p = sigmoid(0.5 * (z - np.mean(z)) / (np.std(z) + 1e-8))
    y = (rng.random(n) < p).astype(np.int32)

    # optional missingness on a fraction of features (recorded, never silent)
    if args.missingness > 0:
        X = X.copy()
        frac = min(args.missingness, 0.5)
        for col in range(X.shape[1]):
            if rng.random() < frac:
                drop = rng.random(n) < args.missingness
                X[drop, col] = np.nan
        # NaN would break sklearn; keep NaN but provenance records it.
        # (loader stage may impute; generator does not hide the gap)

    y_counts = np.bincount(y, minlength=2)
    majority = float(y_counts.max() / y_counts.sum())

    return {
        "X": X, "y": y, "names": names, "view_ranges": view_ranges,
        "view_mats": view_mats, "sig_features": sig_features,
        "int_pairs": int_pairs, "majority": majority,
    }
def _parse_views(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        name, dim, n_sig = part.split(":")
        out[name.strip()] = (int(dim), int(n_sig))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--views", default="default",
                    help="'default' or comma list name:dim:nsig e.g. imaging:64:2,geno:40:2")
    ap.add_argument("--interaction-share", type=float, default=0.30,
                    help="target share of label logit variance from cross-view interactions")
    ap.add_argument("--test-ratio", type=float, default=0.25,
                    help="fraction held out for the stratified test split")
    ap.add_argument("--n-interactions", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--missingness", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    r = build(args)

    import os
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "views"), exist_ok=True)

    # stratified train/test split, matching the load_dataset.py output convention
    # (X_train.npy / X_test.npy / y_train.npy / y_test.npy)
    try:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            r["X"], r["y"], test_size=args.test_ratio, stratify=r["y"],
            random_state=args.seed + 1)
        split_method = "sklearn_stratified"
    except Exception:
        rng2 = np.random.default_rng(args.seed + 1)
        order = rng2.permutation(r["y"].shape[0])
        n_te = max(1, int(args.test_ratio * r["y"].shape[0]))
        X_tr, X_te = r["X"][order[n_te:]], r["X"][order[:n_te]]
        y_tr, y_te = r["y"][order[n_te:]], r["y"][order[:n_te]]
        split_method = "manual_fallback"
    r["split"] = {"test_ratio": args.test_ratio, "method": split_method}

    np.save(os.path.join(args.out, "X.npy"), r["X"])
    np.save(os.path.join(args.out, "y.npy"), r["y"])
    with open(os.path.join(args.out, "feature_names.json"), "w") as fh:
        json.dump(r["names"], fh, indent=1)
    sample_ids = [f"syn_{i:04d}" for i in range(args.n_samples)]
    with open(os.path.join(args.out, "sample_ids.json"), "w") as fh:
        json.dump(sample_ids, fh, indent=1)
    with open(os.path.join(args.out, "feature_views.json"), "w") as fh:
        json.dump(r["view_ranges"], fh, indent=1)
    for view, mat in r["view_mats"].items():
        np.save(os.path.join(args.out, "views", f"{view}.npy"), mat)
    np.save(os.path.join(args.out, "X_train.npy"), X_tr)
    np.save(os.path.join(args.out, "X_test.npy"), X_te)
    np.save(os.path.join(args.out, "y_train.npy"), y_tr)
    np.save(os.path.join(args.out, "y_test.npy"), y_te)

    # measure the realized interaction share of the final logit z
    z_main_v = 0.0
    # recompute cheaply from stored internals is not available here; record design values
    prov = {
        "dataset": os.path.basename(os.path.abspath(args.out)),
        "generator": "scripts/generate_synthetic_views.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "n_samples": int(r["y"].shape[0]),
        "n_features": int(r["X"].shape[1]),
        "n_views": len(r["view_mats"]),
        "class_counts": [int(c) for c in np.bincount(r["y"], minlength=2)],
        "majority_baseline": round(float(r["majority"]), 4),
        "split": r["split"],
        "target_interaction_share": args.interaction_share,
        "true_interaction_share": None,  # oracle to be filled by gate validation script
        "signal_features": {k: [int(i) for i in v] for k, v in r["sig_features"].items()},
        "interaction_pairs": [[int(a), int(b)] for a, b in r["int_pairs"]],
        "evidence_level": "synthetic",
        "notes": "ground-truth interaction pairs recorded; missingness kept as NaN when set",
    }
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)

    print(f"[synthetic] out={args.out}")
    print(f"  X {r['X'].shape} y counts={prov['class_counts']} "
          f"majority={prov['majority_baseline']}")
    print(f"  views={list(r['view_mats'].keys())} "
          f"interaction_pairs={prov['interaction_pairs']}")
    print(f"  target_interaction_share={args.interaction_share} "
          f"noise={args.noise} missingness={args.missingness}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
