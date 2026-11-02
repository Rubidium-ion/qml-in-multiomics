#!/usr/bin/env python3
"""run_benchmark.py — load a v0 dataset dir, run enabled algorithms, emit metrics JSON.

Usage:
    python scripts/run_benchmark.py \
        --data-dir data/tcga_brca_v1 \
        --algos qsvm_angle svm_rbf rf_baseline \
        --out benchmarks/results/run_001.json
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_split(data_dir):
    def L(n):
        return np.load(os.path.join(data_dir, n))
    return L("X_train.npy"), L("X_test.npy"), L("y_train.npy"), L("y_test.npy")


def load_algo_config(name):
    import yaml
    cfg_path = os.path.join(REPO, "algorithms", name, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh) or {}
        return {"params": raw.get("params", {}), **raw}
    return {"params": {}}


def metrics(y_true, y_pred, y_proba=None):
    """Binary and multiclass safe.

    Subtype labels (PAM50: LumA/LumB/Basal/Her2/Normal) are multiclass and very
    imbalanced, so:
      - f1/precision/recall are reported BOTH macro and weighted; macro is the
        honest one because it refuses to hide a dead minority class.
      - accuracy alone is meaningless at ~50% LumA prevalence; balanced_accuracy
        and the majority baseline are always emitted alongside.
      - AUC uses ovr-macro for multiclass, which needs a full (n, n_classes)
        proba matrix.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score, cohen_kappa_score)

    classes = np.unique(y_true)
    n_cls = len(classes)
    binary = n_cls <= 2

    avg = "binary" if binary else "macro"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "n_classes": int(n_cls),
    }
    if not binary:
        out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted",
                                            zero_division=0))
        out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
        # per-class recall exposes classes the model never predicts
        pc = recall_score(y_true, y_pred, average=None, labels=classes,
                          zero_division=0)
        out["min_class_recall"] = float(np.min(pc))
        out["dead_classes"] = int(np.sum(pc == 0))

    out["auc_roc"] = None
    try:
        if y_proba is not None and n_cls > 1:
            P = np.asarray(y_proba, dtype=float)
            if binary:
                col = P[:, 1] if P.ndim == 2 and P.shape[1] >= 2 else P.ravel()
                out["auc_roc"] = float(roc_auc_score(y_true, col))
            elif P.ndim == 2 and P.shape[1] == n_cls:
                out["auc_roc"] = float(roc_auc_score(
                    y_true, P, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc"] = None
    return out


PRE_PROJECTED_WARNING = """
!! {d}
   This directory holds features that were already dimensionality-reduced by
   load_dataset.py. Pooling X_train+X_test and re-splitting them means every CV
   fold is scored on a projection basis that saw part of that fold.
   Measured inflation on tcga_brca_v4: +0.29 accuracy (see
   benchmarks/results/cv_leakage_check.json). Use --cv holdout, or point
   --data-dir at raw features and let each algorithm reduce inside train().
"""


def _looks_decorrelated(M):
    """PCA output is centred and has ~diagonal covariance in its own basis."""
    if M.shape[0] < 3 or M.shape[1] < 2:
        return False
    C = np.cov(M.T)
    dg = float(np.median(np.abs(np.diag(C))))
    if dg <= 0:
        return False
    off = float(np.abs(C[~np.eye(C.shape[0], dtype=bool)]).max())
    centred = float(np.abs(M.mean(axis=0)).max()) < 1e-3 * max(1.0, dg ** 0.5)
    return bool(centred and off / dg < 1e-2)


def _is_pre_projected(data_dir, X_tr, X_te):
    """True if features were already reduced by load_dataset.py.

    Two signatures, both fatal for re-splitting:
      - fitted on train only  -> X_train is centred + decorrelated (v4)
      - fitted on pooled set  -> the STACK is centred + decorrelated (v1, leaky)
    """
    meta = os.path.join(data_dir, "feature_meta.json")
    if os.path.exists(meta):
        try:
            with open(meta) as fh:
                if (json.load(fh) or {}).get("pca_components"):
                    return True
        except Exception:
            pass
    if X_tr.shape[1] >= X_tr.shape[0]:
        return False
    return _looks_decorrelated(X_tr) or _looks_decorrelated(np.vstack([X_tr, X_te]))


def _aggregate(fold_metrics):
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [d[k] for d in fold_metrics if d.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out["std"] = {}
    for k in keys:
        vals = [d[k] for d in fold_metrics if d.get(k) is not None]
        out["std"][k] = float(np.std(vals)) if vals else None
    return out


def _fit_eval(mod, cfg, X_tr, y_tr, X_te, y_te):
    algo = mod.Algorithm(cfg)
    t0 = time.perf_counter()
    algo.train(X_tr, y_tr)
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = algo.predict(X_te)
    t_pred = time.perf_counter() - t1
    proba = algo.predict_proba(X_te) if hasattr(algo, "predict_proba") else None
    m = metrics(y_te, y_pred, proba)
    m["train_time"] = t_train
    m["inference_time"] = t_pred
    return m, algo


def run(args):
    X_tr, X_te, y_tr, y_te = load_split(args.data_dir)
    y_all = np.concatenate([y_tr, y_te])
    counts = np.bincount(y_all)
    majority = float(counts.max() / counts.sum())

    pre_projected = _is_pre_projected(args.data_dir, X_tr, X_te)
    cv_mode = args.cv
    if cv_mode == "auto":
        cv_mode = "holdout" if pre_projected else "kfold"

    if pre_projected and cv_mode == "kfold" and not args.allow_leaky_cv:
        print(PRE_PROJECTED_WARNING.format(d=os.path.abspath(args.data_dir)))
        raise SystemExit(
            "refusing to run k-fold CV on pre-projected features; "
            "pass --allow-leaky-cv if you really want the inflated number"
        )

    print(f"[run_benchmark] n={len(y_all)} labels={counts.tolist()} "
          f"majority_baseline={majority:.3f} mode={cv_mode}"
          + ("  (features look pre-projected)" if pre_projected else ""))

    results = {}
    for name in args.algos:
        cfg = load_algo_config(name)
        mod = importlib.import_module(f"algorithms.{name}")

        if cv_mode == "holdout":
            m, algo = _fit_eval(mod, cfg, X_tr, y_tr, X_te, y_te)
            m["n_splits"] = 1
            m["n_train"] = int(len(y_tr))
            m["n_test"] = int(len(y_te))
        else:
            from sklearn.model_selection import StratifiedKFold
            X_all = np.vstack([X_tr, X_te])
            folds = []
            algo = None
            for rep in range(args.repeats):
                skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                                      random_state=args.seed + rep)
                for tr, te in skf.split(X_all, y_all):
                    fm, algo = _fit_eval(mod, cfg, X_all[tr], y_all[tr],
                                         X_all[te], y_all[te])
                    folds.append(fm)
            m = _aggregate(folds)
            m["n_splits"] = args.n_splits
            m["repeats"] = args.repeats

        m["quantum"] = bool(cfg.get("quantum", False))
        m["fallback"] = bool(getattr(algo, "_used_fallback", False))
        m["majority_baseline"] = majority
        m["above_majority"] = bool(m["accuracy"] > majority)
        results[name] = m

        sd = m.get("std", {}).get("accuracy")
        sd_s = f" +-{sd:.3f}" if sd is not None else ""
        flag = "" if m["above_majority"] else "  <= MAJORITY BASELINE"
        auc = m.get("auc_roc")
        auc_s = f"{auc:.3f}" if auc is not None else "n/a"
        bal = m.get("balanced_accuracy")
        bal_s = f" bal={bal:.3f}" if bal is not None else ""
        dead = m.get("dead_classes")
        dead_s = f"  {int(dead)} DEAD CLASS(ES)" if dead else ""
        print(f"[{name}] acc={m['accuracy']:.3f}{sd_s}{bal_s} auc={auc_s} "
              f"fallback={m['fallback']}{flag}{dead_s}")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_dir": os.path.abspath(args.data_dir),
        "cv_mode": cv_mode,
        "pre_projected_features": pre_projected,
        "n_samples": int(len(y_all)),
        "n_features": int(X_tr.shape[1]),
        "majority_baseline": majority,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[run_benchmark] wrote {args.out}")
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--algos", nargs="+", default=["vqc_qiskit", "qsvm_angle", "svm_rbf", "rf_baseline"])
    p.add_argument("--out", default="benchmarks/results/run_cv_001.json")
    p.add_argument("--cv", choices=["auto", "holdout", "kfold"], default="auto",
                   help="auto = kfold on raw features, holdout on pre-projected")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-leaky-cv", action="store_true",
                   help="permit k-fold on pre-projected features (inflated)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
