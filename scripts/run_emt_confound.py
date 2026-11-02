#!/usr/bin/env python3
"""run_emt_confound.py -- does the PARP interaction signal survive controlling
for mesenchymal state?

Why (authors, 2026-08-08)
-----------------------
The 5-drug pre-screen passed bio's gate (interaction_fraction 0.4265-0.4592,
all PASS). But in ALL FIVE drugs the top SHAP main effects are migration/EMT
genes -- MYOF, THBS1, LCP1, S100A16, LAPTM5, TGFBI -- not DDR genes. Five
independent compounds converging on the same mesenchymal markers is itself a
result, and it threatens the framing:

    interaction_fraction > 0.4 proves INTERACTION STRUCTURE EXISTS.
    It does not prove SYNTHETIC-LETHAL interaction structure exists.

If PARPi sensitivity in GDSC mostly tracks how mesenchymal a cell line is,
then the "interactions" may be co-expression inside an EMT program with
little to do with HR deficiency. That is the same shape as the PGR ->
PR_Status shortcut, one level more abstract: not a single gene standing in
for the label, but a whole transcriptional program.

Three tests, reported together:

  A. How predictable is the label from EMT score ALONE?
     A high AUC from one scalar means the task is largely an EMT readout.

  B. Does interaction_fraction survive EMT ablation?
     Remove the mesenchymal/epithelial marker panel entirely and re-measure.
     If IF collapses below 0.40, the gate was carried by EMT structure.

  C. Does interaction_fraction survive EMT residualisation?
     Stronger: regress the EMT score out of EVERY feature, then re-measure.
     This removes the shared mesenchymal axis without deleting genes, so
     genuine DDR interaction that merely correlates with EMT can still show.

A/B/C disagreeing is informative, not a failure -- see the printed reading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_parp_prescreen import gbdt_cv, interaction_fraction  # noqa: E402
from load_dataset_v3 import load_whitelist                    # noqa: E402

# Hallmark-EMT / mesenchymal core, plus the movers the pre-screen surfaced.
MES = ["VIM", "CDH2", "FN1", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1",
       "SPARC", "TAGLN", "ACTA2", "COL1A1", "COL1A2", "COL5A2", "SERPINE1",
       "TNC", "ITGB1", "MMP2", "THBS1", "TGFBI", "LCP1", "MYOF", "S100A16",
       "LAPTM5", "IGFBP3"]
EPI = ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "ESRP1", "ESRP2", "GRHL2",
       "CLDN4", "CLDN7"]


def emt_score(X, names):
    """Mesenchymal minus epithelial, each z-scored then averaged."""
    def panel(genes):
        idx = [i for i, n in enumerate(names) if n in set(genes)]
        if not idx:
            return None, []
        Z = X[:, idx]
        mu, sd = Z.mean(0), Z.std(0)
        sd[sd == 0] = 1.0
        return ((Z - mu) / sd).mean(1), [names[i] for i in idx]

    m, mg = panel(MES)
    e, eg = panel(EPI)
    if m is None:
        raise SystemExit("no mesenchymal markers found")
    score = m if e is None else m - e
    return score, mg, eg


def residualise(X, s):
    """Remove the component of every feature explained by scalar s (OLS)."""
    sc = (s - s.mean())
    denom = float(sc @ sc)
    if denom == 0:
        return X.copy()
    beta = (X - X.mean(0)) .T @ sc / denom          # per-feature slope
    return X - np.outer(sc, beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    X = np.load(os.path.join(a.cohort, "X.npy")).astype(float)
    y = np.load(os.path.join(a.cohort, "y.npy")).astype(int)
    names = json.load(open(os.path.join(a.cohort,
                                        "feature_names.json")))["gene_names"]
    prov = json.load(open(os.path.join(a.cohort, "provenance.json")))
    base = json.load(open(os.path.join(a.cohort, "prescreen.json")))
    drug = prov["drug"]

    protect = [g for g in load_whitelist() if g in set(names)]
    s, mg, eg = emt_score(X, names)

    print(f"=== {drug} : EMT confound test ===")
    print(f"cohort {X.shape[0]}x{X.shape[1]}  "
          f"mes markers {len(mg)}  epi markers {len(eg)}")
    print(f"baseline (from prescreen.json): IF {base['interaction_fraction']:.4f}"
          f"  AUC {base['gbdt']['auc']:.4f}")

    # point-biserial correlation of EMT score with the label
    r = float(np.corrcoef(s, y)[0, 1])
    print(f"\ncorr(EMT score, responder) = {r:+.4f}")

    # --- A. EMT score alone -------------------------------------------------
    print("\n[A] label predictability from EMT score ALONE (1 feature)")
    balA, _, aucA, sdA = gbdt_cv(s.reshape(-1, 1), y, ["EMT_score"],
                                 seed=a.seed, top_k=1)
    print(f"    AUC = {aucA:.4f} +/- {sdA:.4f}   (full model "
          f"{base['gbdt']['auc']:.4f}, baseline {base['majority_baseline']:.3f})")
    frac_expl = ((aucA - 0.5) / (base["gbdt"]["auc"] - 0.5)
                 if base["gbdt"]["auc"] > 0.5 else float("nan"))
    print(f"    -> one EMT scalar recovers {frac_expl:.1%} of the full "
          f"model's AUC-above-chance")

    # --- B. delete the EMT panel -------------------------------------------
    print("\n[B] interaction_fraction with EMT panel DELETED")
    drop = set(MES) | set(EPI)
    keep = [i for i, n in enumerate(names) if n not in drop]
    Xb, nb = X[:, keep], [names[i] for i in keep]
    pb = [g for g in protect if g in set(nb)]
    iaB = interaction_fraction(Xb, y, nb, seed=a.seed, protect=pb)
    balB, _, aucB, _ = gbdt_cv(Xb, y, nb, seed=a.seed, protect=pb)
    print(f"    removed {X.shape[1]-len(keep)} marker columns")
    print(f"    IF  = {iaB['interaction_fraction']:.4f} "
          f"(delta {iaB['interaction_fraction']-base['interaction_fraction']:+.4f})")
    print(f"    AUC = {aucB:.4f} (delta {aucB-base['gbdt']['auc']:+.4f})")

    # --- C. residualise EMT out of every feature ---------------------------
    print("\n[C] interaction_fraction with EMT score RESIDUALISED out")
    Xc = residualise(X, s)
    iaC = interaction_fraction(Xc, y, names, seed=a.seed, protect=protect)
    balC, _, aucC, _ = gbdt_cv(Xc, y, names, seed=a.seed, protect=protect)
    print(f"    IF  = {iaC['interaction_fraction']:.4f} "
          f"(delta {iaC['interaction_fraction']-base['interaction_fraction']:+.4f})")
    print(f"    AUC = {aucC:.4f} (delta {aucC-base['gbdt']['auc']:+.4f})")
    print("    top main effects after residualisation:")
    for m in iaC["top_main_effects"][:8]:
        tag = " <-- whitelist" if m["feature"] in set(protect) else ""
        print(f"      {m['feature']:12s} {m['strength']:.5f}{tag}")

    # --- reading ------------------------------------------------------------
    print("\n--- reading ---")
    surv = [iaB["interaction_fraction"] > 0.40, iaC["interaction_fraction"] > 0.40]
    if all(surv):
        print("IF holds above 0.40 under BOTH EMT deletion and residualisation:")
        print("the interaction structure is not merely a mesenchymal program.")
    elif not any(surv):
        print("IF drops below 0.40 under both controls: the gate was carried")
        print("by EMT structure. The synthetic-lethality framing is NOT")
        print("supported by this cohort.")
    else:
        print("Controls DISAGREE. Deletion removes genes; residualisation")
        print("removes a shared axis. Partial survival means the signal is")
        print("entangled with mesenchymal state -- report both, claim neither.")
    if aucA > 0.75:
        print(f"NOTE: a single EMT scalar already reaches AUC {aucA:.4f}. Much")
        print("of what the full model 'knows' is cell-line lineage/state.")

    out = os.path.join(a.cohort, "emt_confound.json")
    json.dump({
        "drug": drug,
        "corr_emt_label": r,
        "baseline": {"interaction_fraction": base["interaction_fraction"],
                     "auc": base["gbdt"]["auc"],
                     "majority_baseline": base["majority_baseline"]},
        "A_emt_score_only": {"auc": aucA, "auc_sd": sdA,
                             "frac_of_auc_above_chance": frac_expl},
        "B_emt_deleted": {"interaction_fraction": iaB["interaction_fraction"],
                          "auc": aucB, "n_removed": X.shape[1] - len(keep)},
        "C_emt_residualised": {"interaction_fraction": iaC["interaction_fraction"],
                               "auc": aucC,
                               "top_main_effects": iaC["top_main_effects"][:15]},
        "mes_markers_used": mg, "epi_markers_used": eg,
    }, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
