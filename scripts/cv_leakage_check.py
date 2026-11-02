#!/usr/bin/env python
"""cv_leakage_check.py — is the new CV harness leaking?

The harness pools X_train+X_test and re-splits with StratifiedKFold. But those
arrays were already PCA-projected using a basis fitted on the ORIGINAL training
rows. Every CV fold therefore evaluates on features whose projection basis saw
part of that fold. This measures how much that inflates the score.

Arm A (leaky, = current harness): CV over the pre-projected 8-dim matrix.
Arm B (clean):                    CV from raw gene counts, PCA inside each fold.

Usage:
    python scripts/cv_leakage_check.py --raw-dir ../data/tcga_brca/2026-07-17 \
        --proj-dir data/tcga_brca_v4 --out benchmarks/results/cv_leakage_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_PCA = 8
SEED = 42


def models():
    return {
        "svm_rbf": lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
        "rf_baseline": lambda: RandomForestClassifier(n_estimators=100,
                                                      random_state=SEED),
    }


def score(y_true, y_pred, y_prob):
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc_roc"] = None
    return out


def cv_scores(X, y, fit_inside_fold: bool, repeats=5):
    """fit_inside_fold=False reproduces the harness (PCA already applied)."""
    accs, aucs = [], []
    for rep in range(repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            for name, make in models().items():
                pass
            # evaluate each model separately below
        for name, make in models().items():
            a, u = [], []
            for tr, te in skf.split(X, y):
                Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
                if fit_inside_fold:
                    pipe = Pipeline([
                        ("var", VarianceThreshold(1e-8)),
                        ("std", StandardScaler()),
                        ("pca", PCA(n_components=N_PCA, random_state=SEED)),
                    ]).fit(Xtr)
                    Xtr, Xte = pipe.transform(Xtr), pipe.transform(Xte)
                else:
                    sc = StandardScaler().fit(Xtr)
                    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                m = make().fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                s = score(yte, m.predict(Xte), p)
                a.append(s["accuracy"])
                if s["auc_roc"] is not None:
                    u.append(s["auc_roc"])
            accs.append((name, float(np.mean(a)), float(np.std(a))))
            aucs.append((name, float(np.mean(u)) if u else None))
    agg = {}
    for name in models():
        ac = [m for n, m, _ in accs if n == name]
        sd = [s for n, _, s in accs if n == name]
        au = [m for n, m in aucs if n == name and m is not None]
        agg[name] = {
            "accuracy_mean": float(np.mean(ac)),
            "accuracy_std_across_reps": float(np.std(ac)),
            "accuracy_std_within_cv": float(np.mean(sd)),
            "auc_mean": float(np.mean(au)) if au else None,
        }
    return agg


def load_raw(raw_dir, y):
    files = sorted(glob.glob(os.path.join(raw_dir, "*star_gene_counts.tsv")))
    if not files:
        raise SystemExit(f"no count files in {raw_dir}")
    cols = []
    for f in files:
        v = []
        with open(f) as fh:
            for line in fh:
                if line.startswith(("#", "gene_id")) or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) > 3 and p[2] == "protein_coding":
                    v.append(float(p[3]))
        cols.append(v)
    n = min(len(c) for c in cols)
    X = np.array([c[:n] for c in cols], dtype=float)
    X = np.log1p(X)
    var = X.var(axis=0)
    keep = np.argsort(var)[::-1][:5000]
    return X[:, np.sort(keep)], len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="../data/tcga_brca/2026-07-17")
    ap.add_argument("--proj-dir", default="data/tcga_brca_v4")
    ap.add_argument("--out", default="benchmarks/results/cv_leakage_check.json")
    a = ap.parse_args()

    y = np.concatenate([np.load(f"{a.proj_dir}/y_train.npy"),
                        np.load(f"{a.proj_dir}/y_test.npy")])
    Xp = np.vstack([np.load(f"{a.proj_dir}/X_train.npy"),
                    np.load(f"{a.proj_dir}/X_test.npy")])

    print(f"n={len(y)}  labels={np.bincount(y).tolist()}  "
          f"majority={max(np.bincount(y))/len(y):.3f}")

    print("\nArm A — CV on pre-projected features (what the harness does now)")
    arm_a = cv_scores(Xp, y, fit_inside_fold=False)
    for k, v in arm_a.items():
        print(f"  {k:<12} acc {v['accuracy_mean']:.3f} "
              f"(+-{v['accuracy_std_within_cv']:.3f})  auc {v['auc_mean']:.3f}")

    print("\nArm B — CV from raw counts, PCA fitted inside each fold")
    Xr, n_files = load_raw(a.raw_dir, y)
    print(f"  raw matrix {Xr.shape} from {n_files} files")
    if Xr.shape[0] != len(y):
        print(f"  !! raw rows {Xr.shape[0]} != labels {len(y)} — skipping arm B")
        arm_b = None
    else:
        arm_b = cv_scores(Xr, y, fit_inside_fold=True)
        for k, v in arm_b.items():
            print(f"  {k:<12} acc {v['accuracy_mean']:.3f} "
                  f"(+-{v['accuracy_std_within_cv']:.3f})  auc {v['auc_mean']:.3f}")

    if arm_b:
        print("\nInflation (A - B):")
        for k in arm_a:
            d = arm_a[k]["accuracy_mean"] - arm_b[k]["accuracy_mean"]
            print(f"  {k:<12} {d:+.3f} accuracy")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "author": "authors",
        "n_samples": int(len(y)),
        "majority_baseline": float(max(np.bincount(y)) / len(y)),
        "arm_a_preprojected_cv": arm_a,
        "arm_b_pca_inside_fold": arm_b,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
