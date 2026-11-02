"""Cross-validated experiment runner.

Design commitments, each one a lesson from the GDSC work:

  * feature selection is refit *inside* every training fold. Global selection
    inflated a TCGA cohort from AUC 0.438 to 0.971 — a swing far larger than any
    quantum effect we have ever measured.
  * normalisation is fitted on training rows only, per modality block.
  * quantum arms always emit encoding scale + kernel off-diagonal mass.
  * the majority-class baseline is reported next to every metric, because
    accuracy at 75% class imbalance is not a result.
  * group-aware CV is available and, for grouped cohorts, mandatory to interpret.
    See qmlmo/splits.py for why CPTAC-pooled, CAMELYON17 and OASIS-3 each break
    under a plain random split.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .integrate import BlockNormalizer, FusedDataset
from .models import build_model
from .splits import audit_group_label_balance, build_splits


@dataclass
class RunConfig:
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_splits: int = 5
    seed: int = 42
    normalize: str = "zscore"
    encoding_scale: float | None = None
    select_k: int | None = None
    selector: str = "gbdt"
    protected_features: list[str] = field(default_factory=list)
    #: Minimum / exact number of selected features per modality. Keys match
    #: feature-name prefixes, e.g. {"imaging_mr": 5, "transcriptomics": 5}.
    modality_quotas: dict[str, int] = field(default_factory=dict)
    #: "stratified" | "grouped" | "leave_one_group_out"
    cv: str = "stratified"
    #: labels-table column holding the group id (cohort / centre / subject)
    group_column: str | None = None


def _metrics(y_true, y_pred, proba) -> dict[str, Any]:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, roc_auc_score)
    y_true = np.asarray(y_true)
    n_cls = len(np.unique(y_true))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred,
                             average="binary" if n_cls <= 2 else "macro",
                             zero_division=0)),
        "auc": None,
        "n_test": int(y_true.size),
        "test_positive_rate": float(np.mean(y_true)) if n_cls else None,
    }
    # A single-class test fold has no defined AUC. Under leave-one-group-out this
    # happens for real (a centre with no positives), so it must be recorded as
    # None and excluded from the mean, never silently coerced to 0.5.
    if n_cls < 2:
        out["auc_undefined_reason"] = "single-class test fold"
        return out
    try:
        if proba is not None:
            P = np.asarray(proba, dtype=float)
            if n_cls <= 2:
                col = P[:, 1] if P.ndim == 2 and P.shape[1] >= 2 else P.ravel()
                out["auc"] = float(roc_auc_score(y_true, col))
            else:
                out["auc"] = float(roc_auc_score(y_true, P, multi_class="ovr",
                                                 average="macro"))
    except Exception:
        out["auc"] = None
    return out


def _select_features(X_tr, y_tr, names: Sequence[str], k: int, selector: str,
                     protected: Sequence[str], seed: int,
                     modality_quotas: dict[str, int] | None = None) -> np.ndarray:
    """Fit a selector on TRAINING ROWS ONLY and return kept column indices."""
    keep_forced = [i for i, n in enumerate(names) if n in set(protected)]
    budget = max(0, k - len(keep_forced))

    if selector == "gbdt":
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
        m.fit(X_tr, y_tr)
        score = m.feature_importances_
    elif selector == "variance":
        score = np.nanvar(X_tr, axis=0)
    elif selector == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        score = mutual_info_classif(X_tr, y_tr, random_state=seed)
    else:
        raise ValueError(f"unknown selector {selector!r}")

    forced = set(keep_forced)
    picked = list(keep_forced)
    quotas = modality_quotas or {}
    if sum(quotas.values()) > k:
        raise ValueError("sum of modality_quotas cannot exceed select_k")

    # Score selection is still fitted only on each training fold. Quotas stop a
    # high-dimensional block from silently turning a fusion experiment unimodal.
    for modality, quota in quotas.items():
        candidates = [i for i in np.argsort(score)[::-1]
                      if i not in forced and _modality_of(names[i]) == modality]
        if len(candidates) < quota:
            raise ValueError(
                f"modality {modality!r} has {len(candidates)} selectable features, "
                f"fewer than its requested quota {quota}")
        chosen = candidates[:quota]
        picked.extend(chosen)
        forced.update(chosen)

    remaining = k - len(picked)
    order = [i for i in np.argsort(score)[::-1] if i not in forced]
    return np.asarray(sorted(picked + order[:remaining]), dtype=int)


def _modality_of(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else "?"


def run_cv(dataset: FusedDataset, y: np.ndarray, cfg: RunConfig,
           groups: Sequence[Any] | None = None) -> dict[str, Any]:
    X = np.asarray(dataset.X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}")
    if groups is not None and len(groups) != y.shape[0]:
        raise ValueError(f"{len(groups)} group labels for {y.shape[0]} samples")

    counts = np.bincount(y.astype(int))
    majority = float(counts.max() / counts.sum())

    plan = build_splits(y, cfg.cv, cfg.n_splits, cfg.seed, groups)

    per_model: dict[str, list[dict]] = {name: [] for name in cfg.models}
    reports: dict[str, list[dict]] = {name: [] for name in cfg.models}
    selected_per_fold: list[list[str]] = []

    for fold, tr, te in plan:
        Xtr_raw, Xte_raw = X[tr], X[te]

        # per-block normalisation, fitted on training rows only
        Xtr = np.empty_like(Xtr_raw)
        Xte = np.empty_like(Xte_raw)
        for _mod, (a, b) in dataset.block_slices.items():
            nz = BlockNormalizer(method=cfg.normalize)
            Xtr[:, a:b] = nz.fit_transform(Xtr_raw[:, a:b])
            Xte[:, a:b] = nz.transform(Xte_raw[:, a:b])

        cols = np.arange(X.shape[1])
        if cfg.select_k is not None and cfg.select_k < X.shape[1]:
            cols = _select_features(Xtr, y[tr], dataset.feature_names, cfg.select_k,
                                    cfg.selector, cfg.protected_features, cfg.seed,
                                    cfg.modality_quotas)
        selected_per_fold.append([dataset.feature_names[i] for i in cols])

        for name, params in cfg.models.items():
            p = dict(params)
            if cfg.encoding_scale is not None:
                p.setdefault("encoding_scale", cfg.encoding_scale)
            p.setdefault("seed", cfg.seed)

            model = build_model(name, **p)
            t0 = time.perf_counter()
            model.fit(Xtr[:, cols], y[tr])
            fit_s = time.perf_counter() - t0

            pred = model.predict(Xte[:, cols])
            proba = model.predict_proba(Xte[:, cols])
            m = _metrics(y[te], pred, proba)
            m["fit_seconds"] = round(fit_s, 4)
            m["fold"] = fold
            m["held_out"] = plan.fold_labels[fold]
            m["n_train"] = int(len(tr))
            per_model[name].append(m)
            if model.report:
                reports[name].append({"fold": fold,
                                      "held_out": plan.fold_labels[fold],
                                      **model.report})

    results: dict[str, Any] = {}
    for name, folds in per_model.items():
        agg: dict[str, Any] = {}
        for key in ("accuracy", "balanced_accuracy", "f1", "auc"):
            vals = [f[key] for f in folds if f.get(key) is not None]
            agg[key] = float(np.mean(vals)) if vals else None
            agg[f"{key}_sd"] = float(np.std(vals)) if vals else None
        agg["auc_folds"] = [f["auc"] for f in folds]
        agg["n_folds_scored"] = int(sum(1 for f in folds if f.get("auc") is not None))
        if agg["n_folds_scored"] < len(folds):
            agg["folds_without_auc"] = [
                f["held_out"] for f in folds if f.get("auc") is None]

        # per-group performance: a model that only works on one group is a group
        # detector, and the pooled mean hides that completely
        if plan.kind != "stratified":
            agg["per_group"] = {
                f["held_out"]: {"auc": f["auc"],
                                "balanced_accuracy": f["balanced_accuracy"],
                                "n_test": f["n_test"]}
                for f in folds}

        agg["quantum"] = bool(reports.get(name))
        if reports.get(name):
            agg["quantum_diagnostics"] = reports[name]
            om = [r["offdiag_mean"] for r in reports[name] if "offdiag_mean" in r]
            if om:
                agg["kernel_offdiag_mean"] = float(np.mean(om))
                agg["encoding_scale"] = reports[name][0].get("encoding_scale")
                agg["degenerate_folds"] = int(sum(
                    1 for r in reports[name] if r.get("degenerate")))
        results[name] = agg

    n_common = len(set.intersection(*(set(s) for s in selected_per_fold))) \
        if selected_per_fold else 0

    # which modality actually supplied the selected features. If a "multi-modal"
    # run selects everything from one block, the multi-modal claim is unsupported.
    modality_use: dict[str, int] = {}
    for sel in selected_per_fold:
        for f in sel:
            modality_use[_modality_of(f)] = modality_use.get(_modality_of(f), 0) + 1

    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": dataset.summary(),
        "config": {
            "models": cfg.models, "n_splits": cfg.n_splits, "seed": cfg.seed,
            "normalize": cfg.normalize, "encoding_scale": cfg.encoding_scale,
            "select_k": cfg.select_k, "selector": cfg.selector,
            "protected_features": cfg.protected_features,
            "modality_quotas": cfg.modality_quotas,
            "cv": cfg.cv, "group_column": cfg.group_column,
        },
        "cv_plan": plan.summary(),
        "labels": {"n": int(y.size), "counts": counts.tolist(),
                   "majority_baseline": majority},
        "results": results,
        "selection": {
            "refit_per_fold": True,
            "n_features_per_fold": [len(s) for s in selected_per_fold],
            "n_common_across_folds": n_common,
            "modality_selection_counts": modality_use,
            "genes_per_fold": selected_per_fold if (cfg.select_k or 0) <= 64 else "omitted",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    if groups is not None:
        payload["group_audit"] = audit_group_label_balance(y, groups)

    if cfg.cv == "stratified" and groups is not None:
        payload["interpretation_warning"] = (
            "group labels were supplied but cv='stratified' was used. The reported "
            "scores are an UPPER BOUND: they may reflect group identity rather than "
            "the biological signal. Re-run with cv='leave_one_group_out' before "
            "reporting these numbers."
        )
    return payload


def save_results(payload: dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return os.path.abspath(path)
