#!/usr/bin/env python3
"""negctrl_mannwhitney.py -- Protocol v2.1 final statistics (authors, 2026-08-08).

Collects every prescreen.json on disk, splits drugs into the PARP arm and the
negative-control arm, and runs a two-sided Mann-Whitney U test on
interaction_fraction.

WHY NON-PARAMETRIC
------------------
n is small (5 vs ~14) and IF is a bounded ratio in [0,1]; normality is not
assessable at this n, so a rank test is the honest choice. We report the exact
(permutation-free) U p-value scipy computes for small samples.

WHAT THIS SCRIPT REFUSES TO DO
------------------------------
- It does not drop drugs for having low IF. Every screened drug enters its arm.
- It does not tune the 0.40 gate to make arms separate.
- It does not report one-sided p unless asked, and labels it if so.
A p-value is an observation, not a target. If p > 0.01 after expansion, that is
the result.

Effect size is reported as rank-biserial correlation, which is the effect size
that actually corresponds to Mann-Whitney U (Cohen's d would assume the
normality we just declined to assume).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PARP arm: the five drugs the QML claim was built on.
PARP = {"olaparib", "rucaparib", "niraparib", "veliparib", "talazoparib"}

# Mechanism labels for the negative-control arm.
MECHANISM = {
    "vinblastine": "microtubule",
    "vinorelbine": "microtubule",
    "docetaxel": "microtubule",
    "paclitaxel": "microtubule",
    "methotrexate": "antimetabolite",
    "cytarabine": "antimetabolite",
    "gemcitabine": "antimetabolite",
    "5fluorouracil": "antimetabolite",
    "cisplatin": "alkylating/platinum",
    "oxaliplatin": "alkylating/platinum",
    "temozolomide": "alkylating",
    "bleomycin": "DNA-cleaving",
    "irinotecan": "topoisomerase-I",
    "camptothecin": "topoisomerase-I",
    "teniposide": "topoisomerase-II",
    "bortezomib": "proteasome",
    "palbociclib": "CDK4/6",
}

# Drugs Bio flagged as boundary cases -- kept IN the analysis, but labelled so
# the paper can discuss them rather than quietly benefit from them.
BOUNDARY = {
    "palbociclib": "same indication (breast), different mechanism (CDK4/6)",
    "teniposide": "substituted for Etoposide, which is absent from GDSC2",
    "vinorelbine": "substituted for Vincristine, which is absent from GDSC2",
}

# Requested by Bio but structurally unavailable in GDSC2.
UNAVAILABLE = {
    "Vincristine": "not in GDSC2 (286 drugs screened, no alias)",
    "Carboplatin": "not in GDSC2; no same-class substitute beyond Oxaliplatin",
    "Etoposide": "not in GDSC2; Teniposide substituted (same TOP2 mechanism)",
}


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, "data", "gdsc_*_quantile",
                                           "prescreen.json"))):
        j = json.load(open(p))
        drug = os.path.basename(os.path.dirname(p))[len("gdsc_"):-len("_quantile")]
        prov = j.get("provenance", {})
        binz = prov.get("binarization", {})
        rows.append({
            "drug": drug,
            "arm": "PARP" if drug in PARP else "negative_control",
            "mechanism": "PARP inhibitor" if drug in PARP
                         else MECHANISM.get(drug, "unclassified"),
            "interaction_fraction": j["interaction_fraction"],
            "gbdt_auc": j["gbdt"]["auc"],
            "gbdt_auc_sd": j["gbdt"]["auc_sd"],
            "balanced_accuracy": j["gbdt"]["balanced_accuracy"],
            "n_samples": j["n_samples"],
            "majority_baseline": j["majority_baseline"],
            "binarization": binz,
            "passes_gate": j["interaction_fraction"] > 0.40,
            "boundary_note": BOUNDARY.get(drug),
        })
    return rows


def rank_biserial(a, b, U):
    return 2.0 * U / (len(a) * len(b)) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "data",
                                                  "protocol_v21_final.json"))
    a = ap.parse_args()

    rows = collect()
    parp = [r for r in rows if r["arm"] == "PARP"]
    neg = [r for r in rows if r["arm"] == "negative_control"]

    # Sanity: per-drug binarization must be identical across arms, else the
    # comparison is confounded by label construction rather than biology.
    bins = {json.dumps(r["binarization"], sort_keys=True) for r in rows}

    pv = np.array([r["interaction_fraction"] for r in parp])
    nv = np.array([r["interaction_fraction"] for r in neg])

    print(f"PARP arm             n={len(pv)}  IF mean={pv.mean():.4f} "
          f"median={np.median(pv):.4f} range=[{pv.min():.4f},{pv.max():.4f}]")
    print(f"negative-control arm n={len(nv)}  IF mean={nv.mean():.4f} "
          f"median={np.median(nv):.4f} range=[{nv.min():.4f},{nv.max():.4f}]")
    print()

    # Two-sided is the honest default: we did not pre-register a direction for
    # the negative controls, and several of them screened ABOVE the PARP arm.
    U2, p2 = stats.mannwhitneyu(nv, pv, alternative="two-sided")
    Ug, pg = stats.mannwhitneyu(nv, pv, alternative="greater")
    rb = rank_biserial(nv, pv, U2)

    print(f"Mann-Whitney U (two-sided)  U={U2:.1f}  p={p2:.4f}")
    print(f"Mann-Whitney U (neg>PARP)   U={Ug:.1f}  p={pg:.4f}")
    print(f"rank-biserial r             {rb:+.4f}")
    print()

    target_met = bool(p2 < 0.01)
    print(f"target p<0.01 (two-sided): {'MET' if target_met else 'NOT MET'}")
    if not target_met:
        print("  -> reported as observed. Expansion raises power, it does not")
        print("     manufacture an effect. This is the result, not a failure.")
    print()

    below = [r for r in neg if not r["passes_gate"]]
    print(f"negative controls below the 0.40 gate: {len(below)}")
    for r in sorted(below, key=lambda r: r["interaction_fraction"]):
        print(f"  {r['drug']:16s} IF={r['interaction_fraction']:.4f} "
              f"({r['mechanism']})")
    print()

    # ------------------------------------------------------------------
    # CONSTRUCT VALIDITY OF THE GATE ITSELF.
    #
    # The 0.40 gate is only meaningful if it can FAIL something. Two checks:
    #   (a) how many of ALL drugs clear it,
    #   (b) whether interaction_fraction is just a proxy for task difficulty.
    # If IF tracks classical AUC inversely, then "interaction structure" is
    # not measuring interaction -- it is measuring how little signal the GBDT
    # captured, which is the opposite of a reason to spend quantum resources.
    # ------------------------------------------------------------------
    all_if = np.array([r["interaction_fraction"] for r in rows])
    all_auc = np.array([r["gbdt_auc"] for r in rows])
    rho, p_rho = stats.spearmanr(all_if, all_auc)
    n_pass = int((all_if > 0.40).sum())

    print("--- gate construct validity ---")
    print(f"drugs clearing the 0.40 gate: {n_pass}/{len(all_if)} "
          f"({100*n_pass/len(all_if):.0f}%)")
    print(f"IF dynamic range: {all_if.min():.4f}-{all_if.max():.4f} "
          f"(spread {all_if.max()-all_if.min():.4f})")
    print(f"Spearman(IF, classical GBDT AUC) rho={rho:+.3f} p={p_rho:.2e}")
    if rho < -0.5 and p_rho < 0.05:
        print("  !! IF is strongly INVERSELY coupled to classical AUC.")
        print("     IF largely restates task difficulty, not exploitable")
        print("     interaction structure.")
    print()

    result = {
        "protocol": "v2.1 negative-control expansion",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "binarization_uniform": len(bins) == 1,
        "binarization_variants": sorted(bins),
        "n_drugs_total": len(rows),
        "arms": {"PARP": len(pv), "negative_control": len(nv)},
        "statistics": {
            "test": "Mann-Whitney U",
            "metric": "interaction_fraction",
            "two_sided": {"U": float(U2), "p": float(p2)},
            "one_sided_neg_gt_parp": {"U": float(Ug), "p": float(pg)},
            "rank_biserial_r": float(rb),
            "parp": {"n": len(pv), "mean": float(pv.mean()),
                     "median": float(np.median(pv)), "sd": float(pv.std(ddof=1)),
                     "min": float(pv.min()), "max": float(pv.max())},
            "negative_control": {
                "n": len(nv), "mean": float(nv.mean()),
                "median": float(np.median(nv)), "sd": float(nv.std(ddof=1)),
                "min": float(nv.min()), "max": float(nv.max())},
            "target_p_lt_0_01": target_met,
            "interpretation_note":
                "p is reported as observed. The pre-stated goal of 'pushing "
                "p below 0.01' is not a property the analyst can grant; "
                "expansion increases power only.",
        },
        "gate_threshold": 0.40,
        "n_negctrl_below_gate": len(below),
        "gate_construct_validity": {
            "n_drugs_passing_gate": n_pass,
            "n_drugs_total": int(len(all_if)),
            "pass_rate": round(n_pass / len(all_if), 4),
            "if_min": float(all_if.min()),
            "if_max": float(all_if.max()),
            "if_spread": float(all_if.max() - all_if.min()),
            "spearman_if_vs_classical_auc": {
                "rho": float(rho), "p": float(p_rho)},
            "note":
                "A gate that admits 100% of candidates discriminates nothing. "
                "The strong negative rank correlation between "
                "interaction_fraction and classical GBDT AUC indicates IF "
                "largely restates how little signal the classical model "
                "captured, rather than identifying exploitable interaction "
                "structure. This undercuts the admission criterion itself, "
                "independently of the PARP-vs-control comparison.",
        },
        "drugs_requested_but_unavailable": UNAVAILABLE,
        "boundary_cases": BOUNDARY,
        "drugs": sorted(rows, key=lambda r: (r["arm"],
                                             -r["interaction_fraction"])),
    }
    json.dump(result, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
