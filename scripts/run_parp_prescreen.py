#!/usr/bin/env python3
"""run_parp_prescreen.py — QML admission pre-screen for the PARP/GDSC2 task.

Gate (authors/bio, 2026-08-08): only spend Qiskit resources if a task actually has
interaction structure. interaction_fraction > 0.4 is the agreed threshold.

This script reports THREE numbers, not one, because the threshold alone is a
necessary-but-not-sufficient condition:

  1. interaction_fraction  -- is there interaction structure at all?
  2. GBDT AUC              -- has a classical model already consumed it?
  3. BRCA1/2 ablation      -- is the "interaction" just a single-gene shortcut?

(3) is not optional. PARP sensitivity is textbook BRCA1/2-loss synthetic
lethality, which is structurally identical to the PGR -> PR_Status shortcut we
found on 2026-08-08: one gene nearly synonymous with the label. A high
interaction_fraction driven entirely by BRCA1/2 main effects would be the same
failure wearing a different hat. We measure it up front rather than discover it
after publication.

Usage (once authors supplies the data paths):

  .venv/bin/python scripts/run_parp_prescreen.py \\
      --response  /path/GDSC2_fitted_dose_response.csv \\
      --expression /path/cell_line_expression.csv \\
      --drug Olaparib \\
      --binarize quantile --threshold 25

Data (authors, 2026-08-08): unblocked. Build a cohort first with
scripts/build_gdsc_cohort.py, which joins GDSC2 to the Sanger TPM matrix on
SANGER_MODEL_ID, then point this script at that directory with --cohort.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from load_dataset_v3 import load_whitelist   # noqa: E402

SHORTCUT_GENES = ["BRCA1", "BRCA2"]


def _select(Xtr, names, top_k, protect):
    """Variance-rank top_k, but force `protect` symbols in.

    Why this exists (authors, 2026-08-08)
    -----------------------------------
    gbdt_cv and interaction_fraction each applied their OWN variance filter
    (top_k=300 / 120) on top of the already-built feature matrix. That second
    filter deleted every whitelist gene again: measured on the Olaparib cohort,
    BRCA1 ranked 5012, BRCA2 5011, PARP1 5010 -- all far outside 300.

    The visible symptom was a BRCA1/2 ablation delta of EXACTLY 0.0000 on both
    AUC and interaction_fraction. That is not a null result, it is the
    signature of ablating genes the model never saw. The pathway whitelist was
    rescued into the matrix and then silently dropped one layer down.
    """
    v = Xtr.var(axis=0)
    idx = np.argsort(v)[::-1][:min(top_k, Xtr.shape[1])].tolist()
    if protect:
        want = set(protect)
        forced = [i for i, n in enumerate(names) if n in want]
        idx = sorted(set(idx) | set(forced))
    return np.asarray(idx, dtype=int)


def gbdt_cv(X, y, names=None, seed=42, n_splits=5, n_repeats=5,
            top_k=300, protect=None):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    bal, auc = [], []
    for tr, te in cv.split(X, y):
        keep = _select(X[tr], names or [""] * X.shape[1], top_k, protect)
        m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                       max_depth=3, subsample=0.8,
                                       random_state=seed)
        m.fit(X[tr][:, keep], y[tr])
        p = m.predict_proba(X[te][:, keep])[:, 1]
        bal.append(balanced_accuracy_score(y[te], m.predict(X[te][:, keep])))
        auc.append(roc_auc_score(y[te], p))
    return (float(np.mean(bal)), float(np.std(bal)),
            float(np.mean(auc)), float(np.std(auc)))


def interaction_fraction(X, y, names, seed=42, top_k=120, test_size=0.3,
                         protect=None):
    """Exact SHAP interaction decomposition on a held-out split."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import shap

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    keep = _select(Xtr, names, top_k, protect)
    kept = [names[i] for i in keep]

    m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=3, subsample=0.8,
                                   random_state=seed)
    m.fit(Xtr[:, keep], ytr)
    iv = shap.TreeExplainer(m).shap_interaction_values(Xte[:, keep])
    if isinstance(iv, list):
        iv = iv[-1]
    A = np.abs(np.asarray(iv)).mean(axis=0)
    main = float(np.diag(A).sum())
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    inter = float(off.sum())
    frac = inter / (inter + main) if (inter + main) else 0.0

    iu = np.triu_indices(len(A), k=1)
    strength = off[iu] * 2
    order = np.argsort(strength)[::-1][:15]
    pairs = [{"a": kept[iu[0][o]], "b": kept[iu[1][o]],
              "strength": float(strength[o])} for o in order]
    mo = np.argsort(np.diag(A))[::-1][:15]
    mains = [{"feature": kept[i], "strength": float(np.diag(A)[i])} for i in mo]
    return {"interaction_fraction": frac, "main_mass": main,
            "interaction_mass": inter, "top_pairs": pairs,
            "top_main_effects": mains}


def ablate(X, names, genes):
    """Drop the named genes (and obvious paralog columns) from the matrix."""
    drop = {i for i, n in enumerate(names) if n in set(genes)}
    keep = [i for i in range(X.shape[1]) if i not in drop]
    return X[:, keep], [names[i] for i in keep], sorted(
        {names[i] for i in drop})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True,
                    help="directory from build_gdsc_cohort.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    need = ["X.npy", "y.npy", "feature_names.json"]
    for f in need:
        if not os.path.exists(os.path.join(a.cohort, f)):
            raise SystemExit(f"missing {f} in {a.cohort}; "
                             f"run scripts/build_gdsc_cohort.py first")

    X = np.load(os.path.join(a.cohort, "X.npy")).astype(float)
    y = np.load(os.path.join(a.cohort, "y.npy")).astype(int)
    names = json.load(open(os.path.join(a.cohort,
                                        "feature_names.json")))["gene_names"]
    prov_p = os.path.join(a.cohort, "provenance.json")
    prov = json.load(open(prov_p)) if os.path.exists(prov_p) else {}

    pos = int(y.sum())
    base = max(pos, len(y) - pos) / len(y)
    print(f"cohort   : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"drug     : {prov.get('drug','?')}  "
          f"binarize {prov.get('binarization',{})}")
    print(f"labels   : {pos} responder / {len(y)-pos} non "
          f"(majority baseline {base:.3f})")
    print()

    protect = [g for g in load_whitelist() if g in set(names)]
    print(f"protected from inner variance filters: {len(protect)} "
          f"whitelist genes (else they are dropped again at top_k)")
    print()

    print("[1/3] interaction structure ...")
    ia = interaction_fraction(X, y, names, seed=a.seed, protect=protect)
    frac = ia["interaction_fraction"]
    print(f"  interaction_fraction = {frac:.4f}  (gate: > 0.40)")
    print("  top pairs:")
    for p_ in ia["top_pairs"][:5]:
        print(f"    {p_['a']:12s} x {p_['b']:12s}  {p_['strength']:.5f}")
    print("  top main effects:")
    for m in ia["top_main_effects"][:5]:
        print(f"    {m['feature']:12s} {m['strength']:.5f}")
    print()

    print("[2/3] classical GBDT ceiling ...")
    bal, bal_sd, auc, auc_sd = gbdt_cv(X, y, names, seed=a.seed,
                                       protect=protect)
    print(f"  balanced_acc = {bal:.4f} +/- {bal_sd:.4f}")
    print(f"  AUC          = {auc:.4f} +/- {auc_sd:.4f}")
    print()

    print("[3/3] BRCA1/2 shortcut ablation ...")
    Xa, na, dropped = ablate(X, names, SHORTCUT_GENES)
    if dropped:
        prot2 = [g for g in protect if g not in set(dropped)]
        bal2, _, auc2, _ = gbdt_cv(Xa, y, na, seed=a.seed,
                                   protect=prot2)
        print(f"  dropped: {', '.join(dropped)}")
        print(f"  AUC without BRCA1/2 = {auc2:.4f} "
              f"(delta {auc2-auc:+.4f})")
        ia2 = interaction_fraction(Xa, y, na, seed=a.seed,
                                   protect=prot2)
        d_auc = auc2 - auc
        d_if = ia2["interaction_fraction"] - frac
        print(f"  interaction_fraction without BRCA1/2 = "
              f"{ia2['interaction_fraction']:.4f} (delta {d_if:+.4f})")
        if d_auc == 0.0 and d_if == 0.0:
            print("  !! EXACTLY zero delta on both metrics -- this normally "
                  "means the ablated genes never entered the model. Check "
                  "that they survive the inner top_k filters before reading "
                  "this as 'no BRCA shortcut'.")
    else:
        auc2 = None
        ia2 = None
        print("  BRCA1/2 not among features -- nothing to ablate")
    print()

    verdict = ("PASS -- interaction structure present"
               if frac > 0.40 else
               "FAIL -- below 0.40 gate; QML not justified on this task")
    print(f"VERDICT: {verdict}")
    if auc <= base + 0.02:
        print("NOTE: GBDT barely beats the majority baseline -- the task may "
              "carry little signal at all, interaction or otherwise.")

    result = {
        "cohort": os.path.abspath(a.cohort),
        "provenance": prov,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "majority_baseline": round(base, 4),
        "interaction_fraction": frac,
        "top_pairs": ia["top_pairs"],
        "top_main_effects": ia["top_main_effects"],
        "gbdt": {"balanced_accuracy": bal, "balanced_accuracy_sd": bal_sd,
                 "auc": auc, "auc_sd": auc_sd},
        "ablation_brca": {
            "dropped": dropped,
            "auc": auc2,
            "interaction_fraction": ia2["interaction_fraction"] if ia2 else None,
        },
        "gate_threshold": 0.40,
        "verdict": verdict,
    }
    out = a.out or os.path.join(a.cohort, "prescreen.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
