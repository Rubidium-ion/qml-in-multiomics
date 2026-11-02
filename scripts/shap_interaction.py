#!/usr/bin/env python3
"""shap_interaction.py — quantify explicit feature interactions in the GBDT arm.

bio's question (via authors, 2026-08-08): if the hoped-for quantum advantage is
"capturing feature interactions", can a classical GBDT capture them *and name
them*? A quantum kernel cannot -- it yields a similarity number, not an
attribution.

Method
------
Held-out split (not CV): fit on train, compute SHAP interaction values on test.
TreeExplainer gives an exact (n, k, k) tensor for tree ensembles.

  main effect     = diag of the interaction matrix
  interaction     = off-diagonal, doubled (SHAP splits each pair across i,j)
  interaction_frac= sum|offdiag| / sum|all|   <- the headline number

interaction_frac answers bio directly: what share of this model's decision comes
from pairwise interactions rather than additive main effects? If it is small,
the task is mostly additive and there is no interaction gap for a quantum kernel
to exploit in the first place.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TOP_K = 120          # SHAP interaction is O(k^2) per sample
SEED = 42


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    import shap

    X = np.load(f"{REPO}/data/brca_pr_v1/X.npy")
    y = np.load(f"{REPO}/data/brca_pr_v1/y.npy")
    # gene symbols reconstructed from the STAR TSVs by replaying load_dataset's
    # log1p + top-variance filter; verified bit-exact against X.npy.
    names_p = f"{REPO}/data/brca_pr_v1/feature_names.json"
    names = None
    if os.path.exists(names_p):
        nm = json.load(open(names_p))
        names = [f"{a}" if a else b for a, b in
                 zip(nm["gene_names"], nm["gene_ids"])]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    # variance filter fitted on TRAIN only
    v = np.var(Xtr, axis=0)
    keep = np.argsort(v)[::-1][:TOP_K]
    Xtr_k, Xte_k = Xtr[:, keep], Xte[:, keep]
    kept_names = ([names[i] for i in keep] if names and len(names) > max(keep)
                  else [f"f{i}" for i in keep])
    print(f"[shap] gene symbols: {'yes' if names else 'NO -- falling back to f<idx>'}")

    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                     max_depth=3, subsample=0.8,
                                     random_state=SEED)
    clf.fit(Xtr_k, ytr)
    proba = clf.predict_proba(Xte_k)[:, 1]
    bal = balanced_accuracy_score(yte, clf.predict(Xte_k))
    auc = roc_auc_score(yte, proba)

    t0 = time.time()
    ex = shap.TreeExplainer(clf)
    iv = ex.shap_interaction_values(Xte_k)      # (n, k, k)
    if isinstance(iv, list):
        iv = iv[-1]
    iv = np.asarray(iv)
    elapsed = time.time() - t0

    A = np.abs(iv).mean(axis=0)                 # mean |value| over samples
    main_diag = np.diag(A).copy()
    off = A.copy(); np.fill_diagonal(off, 0.0)
    inter_total = off.sum()                     # already counts (i,j)+(j,i)
    main_total = main_diag.sum()
    frac = inter_total / (inter_total + main_total)

    iu = np.triu_indices(len(A), k=1)
    pair_str = off[iu] * 2
    order = np.argsort(pair_str)[::-1][:15]
    pairs = [{"a": kept_names[iu[0][o]], "b": kept_names[iu[1][o]],
              "strength": float(pair_str[o])} for o in order]

    mo = np.argsort(main_diag)[::-1][:15]
    mains = [{"feature": kept_names[i], "strength": float(main_diag[i])}
             for i in mo]

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "shap.TreeExplainer.shap_interaction_values, exact",
        "split": {"train": int(len(ytr)), "test": int(len(yte)),
                  "test_size": 0.3, "stratified": True, "seed": SEED},
        "top_k_features": TOP_K,
        "holdout_balanced_accuracy": float(bal),
        "holdout_auc": float(auc),
        "interaction_fraction": float(frac),
        "main_effect_mass": float(main_total),
        "interaction_mass": float(inter_total),
        "shap_seconds": round(elapsed, 1),
        "top_interaction_pairs": pairs,
        "top_main_effects": mains,
    }
    p = f"{REPO}/benchmarks/results/brca_pr_v1_shap_interaction.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"holdout bal={bal:.3f} auc={auc:.3f}")
    print(f"interaction_fraction = {frac:.4f}  "
          f"(interaction {inter_total:.4f} vs main {main_total:.4f})")
    print(f"shap {elapsed:.1f}s -> {p}")
    for q in pairs[:5]:
        print(f"   {q['a']} x {q['b']}  {q['strength']:.5f}")


if __name__ == "__main__":
    main()
