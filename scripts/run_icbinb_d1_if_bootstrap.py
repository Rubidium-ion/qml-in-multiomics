#!/usr/bin/env python
"""run_icbinb_d1_if_bootstrap.py — D1: full-pipeline cell-line bootstrap CI + threshold scan.

Bio D1 (Bio_statistical_revision_addendum.md), frozen 2026-10-01:
- Every one of the 22 GDSC2 cohorts gets a cell-line bootstrap 95% CI on its
  interaction fraction (IF). Each bootstrap replicate must rerun the FULL
  pipeline on the resampled cell lines: 5-fold CV with fold-local feature
  selection + GBDT fit + SHAP TreeExplainer interaction decomposition.
  We do NOT resample the final IF values.
- IF is computed in BOTH conventions (C1):
    full: sum|offdiag|  (each symmetric pair counted twice — the original
          pre-specified convention the 0.40 gate was calibrated under)
    half: 0.5*sum|offdiag| (each pair once)
- Threshold scan 0.00..1.00 step 0.02 (51 steps) over the 22 per-drug point
  estimates: admitted count, admitted-task classical performance, etc.
  Robustness / operating-characteristic analysis, not a new gate rule.

Pipeline per replicate (identical to the committed C1 pipeline):
- Fold-local feature selection: top-120 variance on the fold TRAINING partition
  plus the 68-gene DDR/HR whitelist (protected), matching the C1 table.
- GBDT: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
  max_depth=3, subsample=0.8, random_state=42) — fixed, same as C1.
- SHAP: shap 0.52.0 TreeExplainer.shap_interaction_values, exact tensor.
- Replicate IF = mean over 5 stratified CV folds of fold-level IF
  (per-fold AUC also collected for the threshold scan's classical-performance
  column; pooled AUC across folds is the drug-level classical estimate).

Bootstrapping: resample n cell lines WITH REPLACEMENT (n = cohort size),
keeping the label structure; each replicate is a self-contained full-pipeline
run. CI = 2.5/97.5 percentiles of the 500-replicate empirical distribution
(ECDF reported).

Outputs (exact requested names):
  benchmarks/results/icbinb_d1_if_bootstrap_ci.{csv,json}
  benchmarks/results/icbinb_d1_threshold_scan.{csv,json}
Per-drug replicate arrays are cached under
  benchmarks/results/icbinb_d1_bootstrap_runs/<drug>.npz  (persist, resumable)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from load_dataset_v3 import load_whitelist  # noqa: E402

N_DRUGS_EXPECTED = 22
GATE = 0.40
TOP_K = 120
N_FOLDS = 5
SEED = 42
N_BOOT_DEFAULT = 500
N_BOOT_MIN = 200

RESULTS = os.path.join(REPO, "benchmarks", "results")
RUNS_DIR = os.path.join(RESULTS, "icbinb_d1_bootstrap_runs_v2")


# ---------------------------------------------------------------------- data

def cohort_dir(drug: str) -> str:
    p = os.path.join(REPO, "data", f"gdsc_{drug}_quantile")
    need = ["X.npy", "y.npy", "feature_names.json"]
    for f in need:
        if not os.path.exists(os.path.join(p, f)):
            raise FileNotFoundError(f"{f} missing for {drug} ({p})")
    return p


def load_cohort(drug: str):
    p = cohort_dir(drug)
    X = np.load(os.path.join(p, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(p, "y.npy")).astype(int)
    names = json.load(open(os.path.join(p, "feature_names.json")))["gene_names"]
    # Stable original cell-line IDs are the cluster labels for the resampling
    # and GroupKFold guard. Never let copies of one original line cross folds.
    sample_ids = json.load(open(os.path.join(p, "sample_ids.json")))
    if len(sample_ids) != len(y) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{drug}: sample_ids must be one unique ID per original row")
    protect = [g for g in load_whitelist() if g in set(names)]
    forced = np.array([i for i, n in enumerate(names) if n in set(protect)], dtype=int)
    return X, y, names, forced, np.asarray(sample_ids, dtype=object)


def drug_frame() -> list[dict]:
    """22-drug sampling frame from the committed C1 table."""
    c1 = json.load(open(os.path.join(RESULTS, "icbinb_c1_shap_convention_IF.json")))
    out = []
    for c in c1["cohorts"]:
        out.append({
            "drug": c["drug"],
            "if_full": c["if_full"],
            "if_half": c["if_half"],
            "gbdt_auc": c["gbdt_auc"],
            "n": c["n"],
            "n_kept": c["n_kept"],
        })
    out.sort(key=lambda r: r["drug"])
    assert len(out) == N_DRUGS_EXPECTED, f"expected {N_DRUGS_EXPECTED} drugs, got {len(out)}"
    return out


# ------------------------------------------------------------------- pipeline

def _select_train(Xtr: np.ndarray, forced: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    """Fold-local feature selection: top-k variance on TRAIN + protected whitelist."""
    v = Xtr.var(axis=0)
    idx = set(np.argsort(v)[::-1][:min(top_k, Xtr.shape[1])].tolist())
    idx.update(forced.tolist())
    return np.sort(np.array(sorted(idx), dtype=int))


def _fold_one(tr, te, X, y, groups, forced, seed):
    """One CV fold: fold-local selection + GBDT fit + SHAP. Returns fold IFs/AUC."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import shap
    overlap = set(groups[tr]).intersection(groups[te])
    if overlap:
        raise RuntimeError(f"duplicate cluster leakage: {len(overlap)} groups cross fold")
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    keep = _select_train(Xtr, forced)
    m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=3, subsample=0.8,
                                   random_state=SEED)
    m.fit(Xtr[:, keep], ytr)
    p = m.predict_proba(Xte[:, keep])[:, 1]
    auc = float(roc_auc_score(yte, p))
    iv = shap.TreeExplainer(m).shap_interaction_values(Xte[:, keep])
    if isinstance(iv, list):
        iv = iv[-1]
    A = np.abs(np.asarray(iv, dtype=np.float64)).mean(axis=0)
    main = float(np.diag(A).sum())
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    inter_full = float(off.sum())
    total = inter_full + main
    if_full = inter_full / total if total > 0 else 0.0
    if_half = (0.5 * inter_full) / (0.5 * inter_full + main) if total > 0 else 0.0
    return (if_full, if_half, auc, len(keep), len(overlap))


def one_replicate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, forced: np.ndarray,
                  seed: int, n_folds_threads: int = N_FOLDS) -> dict:
    """Full pipeline on one bootstrap-resampled cell-line cohort.

    5-fold stratified CV; per fold: fold-local feature selection, GBDT fit,
    SHAP TreeExplainer interaction tensor on the held-out fold; IF (full and
    half) per fold; replicate IF = mean over folds. Folds are run concurrently
    on a small thread pool (GBDT fit releases the GIL) to keep cores busy.
    """
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.model_selection import GroupKFold

    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    if pos == 0 or neg == 0:
        return {"if_full": np.nan, "if_half": np.nan,
                "auc_folds": [np.nan] * N_FOLDS, "auc": np.nan,
                "n_kept": int(len(forced)), "n": n, "degenerate": True}

    # D1 v2 hard requirement: GroupKFold by original cell-line ID. Bootstrap
    # duplicates remain within the same fold, so no duplicate leakage.
    skf = GroupKFold(n_splits=N_FOLDS)
    splits = list(skf.split(X, y, groups=groups))
    n_thr = max(1, min(n_folds_threads, N_FOLDS))
    if n_thr == 1:
        out = [_fold_one(tr, te, X, y, groups, forced, seed) for tr, te in splits]
    else:
        with ThreadPoolExecutor(max_workers=n_thr) as ex:
            out = list(ex.map(lambda s: _fold_one(s[0], s[1], X, y, groups, forced, seed),
                              splits))
    ifs_full = [o[0] for o in out]
    ifs_half = [o[1] for o in out]
    aucs = [o[2] for o in out]
    kepts = [o[3] for o in out]
    overlaps = [o[4] for o in out]
    return {
        "if_full": float(np.nanmean(ifs_full)),
        "if_half": float(np.nanmean(ifs_half)),
        "auc_folds": aucs,
        "auc": float(np.nanmean(aucs)),
        "n_kept": int(np.max(kepts)),
        "n": n,
        "degenerate": False,
        "fold_group_overlap_counts": overlaps,
    }


def drug_bootstrap(drug: str, n_boot: int, boot_seed_base: int = 20261002,
                   log: bool = False, n_threads: int = 2) -> dict:
    """Run n_boot full-pipeline bootstrap replicates for one drug."""
    X, y, names, forced, original_ids = load_cohort(drug)
    # deterministic, independent stream per drug: seed derived from name
    h = int(hashlib.sha256(f"d1|{drug}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(boot_seed_base * 1000003 + h % 1000000)
    n = len(y)
    reps_full, reps_half, aucs, kepts, overlap_max = [], [], [], [], []
    t0 = time.time()
    for b in range(n_boot):
        seed = int(rng.integers(0, 2**31 - 1))
        idx = rng.integers(0, n, size=n)
        r = one_replicate(X[idx], y[idx], original_ids[idx], forced, seed, n_folds_threads=n_threads)
        reps_full.append(r["if_full"])
        reps_half.append(r["if_half"])
        aucs.append(r["auc"])
        kepts.append(r["n_kept"])
        overlap_max.append(max(r.get("fold_group_overlap_counts", [0])))
        if log and (b + 1) % 50 == 0:
            print(f"[{drug}] {b+1}/{n_boot} "
                  f"(mean IF_full={np.nanmean(reps_full):.4f}, "
                  f"{(time.time()-t0)/(b+1):.2f}s/rep)", flush=True)
    return {
        "drug": drug,
        "n": n,
        "n_boot": n_boot,
        "boot_seed_base": boot_seed_base,
        "reps_full": np.asarray(reps_full, dtype=np.float64),
        "reps_half": np.asarray(reps_half, dtype=np.float64),
        "reps_auc": np.asarray(aucs, dtype=np.float64),
        "reps_kept": np.asarray(kepts, dtype=np.int64),
        "max_fold_group_overlap": int(max(overlap_max) if overlap_max else 0),
        "point_if_full": float(np.nanmean(reps_full)),
        "point_if_half": float(np.nanmean(reps_half)),
        "point_auc": float(np.nanmean(aucs)),
        "wall_sec": round(time.time() - t0, 1),
        "elapsed_per_rep_sec": round((time.time() - t0) / n_boot, 2),
    }


# --------------------------------------------------------------- aggregation

def quantile_ci(reps: np.ndarray, lo=0.025, hi=0.975):
    reps = reps[~np.isnan(reps)]
    if len(reps) == 0:
        return np.nan, np.nan
    return float(np.quantile(reps, lo)), float(np.quantile(reps, hi))


def ecdf_stats(reps: np.ndarray):
    reps = reps[~np.isnan(reps)]
    return {
        "n_valid": int(len(reps)),
        "mean": float(np.mean(reps)),
        "sd": float(np.std(reps, ddof=1)),
        "median": float(np.median(reps)),
        "p05": float(np.quantile(reps, 0.05)),
        "p95": float(np.quantile(reps, 0.95)),
    }


def _stability_stats(reps: np.ndarray, point: float) -> dict:
    """Group-bootstrap diagnostics; intervals are sensitivity outputs, not main CIs."""
    x = np.asarray(reps, dtype=float)
    x = x[np.isfinite(x)]
    lo, hi = quantile_ci(x)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return {
        "bootstrap_mean": float(np.mean(x)),
        "bootstrap_median": float(np.median(x)),
        "bootstrap_bias_mean_minus_point": float(np.mean(x) - point),
        "bootstrap_sd": float(np.std(x, ddof=1)),
        "bootstrap_iqr": float(q3 - q1),
        "pr_boot_ge_0_40": float(np.mean(x >= GATE)),
        "percentile_interval_95": [lo, hi],
        # Basic interval is the reflection of the percentile endpoints about
        # the original point estimate; report as a sensitivity interval only.
        "basic_interval_95": [float(2 * point - hi), float(2 * point - lo)],
    }


def drug_summary(drug: str, boot: dict, frame: dict) -> dict:
    lo_f, hi_f = quantile_ci(boot["reps_full"])
    lo_h, hi_h = quantile_ci(boot["reps_half"])
    sf = _stability_stats(boot["reps_full"], frame["if_full"])
    sh = _stability_stats(boot["reps_half"], frame["if_half"])
    return {
        "drug": drug,
        "n_cell_lines": int(boot["n"]),
        "n_bootstrap_replicates": int(boot["n_boot"]),
        "point_estimate_if_full": frame["if_full"],
        "point_estimate_if_half": frame["if_half"],
        "bootstrap_mean_if_full": sf["bootstrap_mean"],
        "bootstrap_median_if_full": sf["bootstrap_median"],
        "bootstrap_bias_if_full": sf["bootstrap_bias_mean_minus_point"],
        "bootstrap_sd_if_full": sf["bootstrap_sd"],
        "bootstrap_iqr_if_full": sf["bootstrap_iqr"],
        "pr_boot_if_full_ge_0_40": sf["pr_boot_ge_0_40"],
        "bootstrap_ci95_if_full": sf["percentile_interval_95"],
        "basic_ci95_if_full": sf["basic_interval_95"],
        "bootstrap_mean_if_half": sh["bootstrap_mean"],
        "bootstrap_median_if_half": sh["bootstrap_median"],
        "bootstrap_bias_if_half": sh["bootstrap_bias_mean_minus_point"],
        "bootstrap_sd_if_half": sh["bootstrap_sd"],
        "bootstrap_iqr_if_half": sh["bootstrap_iqr"],
        "pr_boot_if_half_ge_0_40": sh["pr_boot_ge_0_40"],
        "bootstrap_ci95_if_half": sh["percentile_interval_95"],
        "basic_ci95_if_half": sh["basic_interval_95"],
        "admit_at_0.40_full_point": bool(frame["if_full"] >= GATE),
        "admit_at_0.40_half_point": bool(frame["if_half"] >= GATE),
        "admit_at_0.40_full_ci_low": bool(lo_f >= GATE),
        "admit_at_0.40_half_ci_low": bool(lo_h >= GATE),
        "point_estimate_classical_auc": frame["gbdt_auc"],
        "bootstrap_mean_classical_auc": ecdf_stats(boot["reps_auc"])["mean"],
        "bootstrap_ci95_classical_auc": quantile_ci(boot["reps_auc"]),
        "median_kept_features": float(np.median(boot["reps_kept"])),
        "wall_sec": boot["wall_sec"],
        "max_fold_group_overlap": int(boot.get("max_fold_group_overlap", -1)),
        "ecdf_full": ecdf_stats(boot["reps_full"]),
        "ecdf_half": ecdf_stats(boot["reps_half"]),
    }


# ------------------------------------------------------------- threshold scan

def threshold_scan(frame: list[dict], boot: dict | None = None) -> list[dict]:
    """Scan 0.00..1.00 step 0.02 over the 22 point estimates (both conventions).

    For each threshold report: admitted count; admitted-task classical
    performance (mean AUC of admitted tasks); Spearman rho between IF and
    classical AUC restricted to admitted tasks (where defined). Where
    per-replicate IFs are available, also report the bootstrap distribution
    of the admitted count at that threshold (operating-characteristic CI).
    """
    from scipy.stats import spearmanr
    steps = [round(t, 2) for t in np.arange(0.0, 1.0001, 0.02)]
    rows = []
    for t in steps:
        for conv in ("full", "half"):
            admit = [f for f in frame if f[f"if_{conv}"] >= t]
            n_admit = len(admit)
            if n_admit:
                aucs = [f["gbdt_auc"] for f in admit]
                mean_auc = float(np.mean(aucs))
                # association IF~AUC among admitted (need >=3 & variance)
                if len(admit) >= 3 and len({f[f"if_{conv}"] for f in admit}) > 1 \
                        and len({f["gbdt_auc"] for f in admit}) > 1:
                    rho, p = spearmanr([f[f"if_{conv}"] for f in admit], aucs)
                else:
                    rho, p = np.nan, np.nan
            else:
                mean_auc, rho, p = np.nan, np.nan, np.nan
            # bootstrap distribution of the admitted count at this threshold
            if boot:
                reps = np.stack([b[f"reps_{conv}"] for b in boot.values()
                                if b.get(f"reps_{conv}") is not None
                                and int(b["n_boot"]) > 0])
                adm_b = (reps >= t).astype(int).sum(axis=1)
                boot_adm = {
                    "expected": float(np.mean(adm_b)),
                    "ci95_low": float(np.quantile(adm_b, 0.025)),
                    "ci95_high": float(np.quantile(adm_b, 0.975)),
                }
            else:
                boot_adm = {"expected": np.nan, "ci95_low": np.nan,
                           "ci95_high": np.nan}
            rows.append({
                "threshold": t,
                "convention": conv,
                "n_admitted": n_admit,
                "admitted_drugs": ";".join(sorted(f["drug"] for f in admit)),
                "admitted_mean_classical_auc": mean_auc,
                "if_auc_spearman_rho_admitted": float(rho) if rho == rho else np.nan,
                "if_auc_spearman_p_admitted": float(p) if p == p else np.nan,
                "bootstrap_expected_admitted": boot_adm["expected"],
                "bootstrap_admitted_ci95_low": boot_adm["ci95_low"],
                "bootstrap_admitted_ci95_high": boot_adm["ci95_high"],
                "is_preset_gate": bool(abs(t - GATE) < 1e-9),
                "highlight_0.30_0.50": bool(0.30 <= t <= 0.50),
            })
    return rows


# -------------------------------------------------------------------- runner

_WORKER_THREADS = 2


def _worker(drug: str, n_boot: int, boot_seed_base: int, log: bool,
            n_threads: int = _WORKER_THREADS) -> dict:
    return drug_bootstrap(drug, n_boot, boot_seed_base, log=log,
                          n_threads=n_threads)


def run_all(drugs: list[str], n_boot: int, n_workers: int,
            boot_seed_base: int = 20261002, resume: bool = True) -> dict:
    os.makedirs(RUNS_DIR, exist_ok=True)
    results = {}
    pending = []
    for drug in drugs:
        npz = os.path.join(RUNS_DIR, f"{drug}.npz")
        if resume and os.path.exists(npz):
            z = np.load(npz)
            stored = {
                "drug": drug, "n": int(z["n"]), "n_boot": int(z["n_boot"]),
                "boot_seed_base": int(z["boot_seed_base"]),
                "reps_full": z["reps_full"], "reps_half": z["reps_half"],
                "reps_auc": z["reps_auc"], "reps_kept": z["reps_kept"],
                "wall_sec": float(z["wall_sec"]),
                "max_fold_group_overlap": int(z["max_fold_group_overlap"]) if "max_fold_group_overlap" in z else -1,
            }
            if stored["n_boot"] == n_boot and stored["boot_seed_base"] == boot_seed_base:
                results[drug] = stored
                continue
            print(f"[resume] {drug}: stored n_boot={stored['n_boot']} != {n_boot} "
                  f"or seed mismatch -> recomputing", flush=True)
            pending.append(drug)
            continue
        pending.append(drug)
    if pending:
        t0 = time.time()
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        n_threads = _WORKER_THREADS
        if n_workers <= 1:
            for i, drug in enumerate(pending):
                print(f"=== {drug} ({i+1}/{len(pending)}) ===", flush=True)
                r = _worker(drug, n_boot, boot_seed_base, log=True, n_threads=1)
                _save_runs(drug, r)
                results[drug] = r
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_worker, d, n_boot, boot_seed_base, False,
                                 n_threads): d
                        for d in pending}
                done = 0
                for f in as_completed(futs):
                    d = futs[f]
                    r = f.result()
                    _save_runs(d, r)
                    results[d] = r
                    done += 1
                    print(f"[{done}/{len(pending)}] {d}: "
                          f"IF_full={r['point_if_full']:.4f} "
                          f"IF_half={r['point_if_half']:.4f} "
                          f"({r['wall_sec']:.0f}s)", flush=True)
        print(f"ALL DONE {len(pending)} drugs in {time.time()-t0:.0f}s", flush=True)
    return results


def _save_runs(drug: str, r: dict):
    np.savez_compressed(os.path.join(RUNS_DIR, f"{drug}.npz"),
                        n=np.int64(r["n"]), n_boot=np.int64(r["n_boot"]),
                        boot_seed_base=np.int64(r["boot_seed_base"]),
                        reps_full=r["reps_full"], reps_half=r["reps_half"],
                        reps_auc=r["reps_auc"], reps_kept=r["reps_kept"],
                        wall_sec=np.float64(r["wall_sec"]),
                        max_fold_group_overlap=np.int64(r["max_fold_group_overlap"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--seed-base", type=int, default=20261002)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--drugs", nargs="*", default=None)
    a = ap.parse_args()

    frame = drug_frame()
    drugs = a.drugs or [f["drug"] for f in frame]
    if a.n_boot < N_BOOT_MIN:
        print(f"NOTE: --n-boot={a.n_boot} is below the {N_BOOT_MIN}-replicate "
              f"minimum; honoring the explicit value for this run", flush=True)
    n_boot = a.n_boot
    print(f"D1 bootstrap: {len(drugs)} drugs x {n_boot} full-pipeline replicates "
          f"({N_FOLDS}-fold, fold-local selection+GBDT+SHAP), {a.workers} workers",
          flush=True)

    boot = run_all(drugs, n_boot, a.workers, a.seed_base, resume=not a.no_resume)

    summaries = [drug_summary(f["drug"], boot[f["drug"]], f) for f in frame
                 if f["drug"] in boot]

    # ---- bootstrap CI table ----
    ci_csv = os.path.join(RESULTS, "icbinb_d1_if_bootstrap_ci_v2_groupkfold.csv")
    ci_json = os.path.join(RESULTS, "icbinb_d1_if_bootstrap_ci_v2_groupkfold.json")
    ci_rows = [{
        "drug": s["drug"],
        "n_cell_lines": s["n_cell_lines"],
        "n_bootstrap_replicates": s["n_bootstrap_replicates"],
        "if_full_point": s["point_estimate_if_full"],
        "if_full_ci95_low": s["bootstrap_ci95_if_full"][0],
        "if_full_ci95_high": s["bootstrap_ci95_if_full"][1],
        "if_full_basic_ci95_low": s["basic_ci95_if_full"][0],
        "if_full_basic_ci95_high": s["basic_ci95_if_full"][1],
        "if_full_bootstrap_median": s["bootstrap_median_if_full"],
        "if_full_bootstrap_bias": s["bootstrap_bias_if_full"],
        "if_full_bootstrap_sd": s["bootstrap_sd_if_full"],
        "if_full_bootstrap_iqr": s["bootstrap_iqr_if_full"],
        "pr_boot_if_full_ge_0_40": s["pr_boot_if_full_ge_0_40"],
        "if_half_point": s["point_estimate_if_half"],
        "if_half_ci95_low": s["bootstrap_ci95_if_half"][0],
        "if_half_ci95_high": s["bootstrap_ci95_if_half"][1],
        "if_half_basic_ci95_low": s["basic_ci95_if_half"][0],
        "if_half_basic_ci95_high": s["basic_ci95_if_half"][1],
        "if_half_bootstrap_median": s["bootstrap_median_if_half"],
        "if_half_bootstrap_bias": s["bootstrap_bias_if_half"],
        "if_half_bootstrap_sd": s["bootstrap_sd_if_half"],
        "if_half_bootstrap_iqr": s["bootstrap_iqr_if_half"],
        "pr_boot_if_half_ge_0_40": s["pr_boot_if_half_ge_0_40"],
        "admit_0.40_full_point": s["admit_at_0.40_full_point"],
        "admit_0.40_half_point": s["admit_at_0.40_half_point"],
        "admit_0.40_full_ci_low": s["admit_at_0.40_full_ci_low"],
        "admit_0.40_half_ci_low": s["admit_at_0.40_half_ci_low"],
        "classical_auc_point": s["point_estimate_classical_auc"],
        "classical_auc_ci95_low": s["bootstrap_ci95_classical_auc"][0],
        "classical_auc_ci95_high": s["bootstrap_ci95_classical_auc"][1],
        "median_kept_features": s["median_kept_features"],
        "max_fold_group_overlap": s["max_fold_group_overlap"],
    } for s in summaries]
    with open(ci_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ci_rows[0].keys()))
        w.writeheader()
        for r in ci_rows:
            w.writerow(r)
    ci_meta = {
        "format": "icbinb_d1_if_bootstrap_ci_v2_groupkfold.v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z").strip(),
        "task": "D1 Test 1: cell-line bootstrap 95% CI on interaction fraction "
                "(full-pipeline re-run each replicate: 5-fold fold-local "
                "feature selection + GBDT + SHAP TreeExplainer)",
        "n_bootstrap_replicates": n_boot,
        "n_drugs": len(summaries),
        "boot_seed_base": a.seed_base,
        "resampling": "original cell-line clusters resampled with replacement; duplicate copies retain original cell-line group ID; each replicate reruns full pipeline (not final-IF resampling)",
        "conventions": {
            "full": "sum|offdiag| (each symmetric pair counted twice; the "
                    "pre-specified convention under which the 0.40 gate was "
                    "calibrated)",
            "half": "0.5*sum|offdiag| (each pair once)",
        },
        "ci_method": "Percentile and basic intervals are reported as sensitivity intervals only; main D1 interpretation uses group-resampling stability diagnostics (point, ECDF, median, bias, SD/IQR, Pr_boot(IF>=0.40)).",
        "pipeline": {
            "feature_selection": f"top-{TOP_K} variance on fold train + "
                                 "68-gene DDR/HR whitelist (protected)",
            "gbdt": "GradientBoostingClassifier(300, lr=0.05, depth=3, "
                    "subsample=0.8, seed=42)",
            "shap": "shap 0.52.0 TreeExplainer.shap_interaction_values (exact)",
            "cv": f"{N_FOLDS}-fold GroupKFold by original cell-line ID (duplicate-leakage protected)",
            "if_per_replicate": "mean of per-fold IF over the 5 folds",
        },
        "gate_threshold": GATE,
        "qa": {"fold_train_test_original_cell_line_overlap_required": 0,
               "all_cohort_max_overlap": max(s["max_fold_group_overlap"] for s in summaries),
               "v1_status": "deprecated-leakage: row-bootstrap plus stratified CV allowed duplicate original cell lines across train/test; do not use v1 CIs."},
        "note": "0.40 is the original pre-specified threshold; it is NOT "
                "reinterpreted as a data-derived optimum. CI is on the "
                "bootstrap distribution of per-replicate full-pipeline IFs.",
        "cohorts": summaries,
    }
    json.dump(ci_meta, open(ci_json, "w"), indent=2)
    print(f"wrote {ci_csv}\nwrote {ci_json}", flush=True)

    # ---- threshold scan ----
    scan = threshold_scan(frame, boot)
    scan_csv = os.path.join(RESULTS, "icbinb_d1_threshold_scan.csv")
    scan_json = os.path.join(RESULTS, "icbinb_d1_threshold_scan.json")
    with open(scan_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scan[0].keys()))
        w.writeheader()
        for r in scan:
            w.writerow(r)
    scan_meta = {
        "format": "icbinb_d1_threshold_scan.v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z").strip(),
        "task": "D1: threshold 0.00..1.00 step 0.02 scan over the 22 point "
                "IF estimates (both conventions). Robustness / "
                "operating-characteristic analysis, NOT a new gate-selection "
                "rule. 0.40 flagged as the original pre-specified threshold.",
        "n_drugs": len(frame),
        "steps": len(set(r["threshold"] for r in scan)),
        "highlight_band": [0.30, 0.50],
        "rows": scan,
    }
    json.dump(scan_meta, open(scan_json, "w"), indent=2)
    print(f"wrote {scan_csv}\nwrote {scan_json}", flush=True)

    # console summary
    print("\n=== D1 SUMMARY ===")
    print(f"n_bootstrap={n_boot}  n_drugs={len(summaries)}")
    print("drug                      IFfull   CI95full              IFhalf   CI95half              AUCpoint")
    for s in summaries:
        lo, hi = s["bootstrap_ci95_if_full"]
        loh, hih = s["bootstrap_ci95_if_half"]
        print(f"{s['drug']:24s} {s['point_estimate_if_full']:.4f}  "
              f"[{lo:.4f},{hi:.4f}]  {s['point_estimate_if_half']:.4f}  "
              f"[{loh:.4f},{hih:.4f}]  {s['point_estimate_classical_auc']:.4f}")
    admit_full = sum(1 for s in summaries if s["admit_at_0.40_full_point"])
    admit_half = sum(1 for s in summaries if s["admit_at_0.40_half_point"])
    print(f"\nAt 0.40 (pre-specified gate): full={admit_full}/22 admit, "
          f"half={admit_half}/22 admit (point estimates)")


if __name__ == "__main__":
    main()
