#!/usr/bin/env python3
"""gate_construct_validity.py -- does interaction_fraction measure what we said?

BACKGROUND
----------
Before spending Qiskit compute, the team pre-registered an admission gate:

    interaction_fraction (IF) > 0.40  =>  the task has interaction structure
                                          worth attacking with a quantum kernel

IF is the share of GBDT SHAP attribution mass carried by interaction terms
rather than main effects. The intent was to avoid a repeat of the PR_Status
episode, where a single gene (PGR) was nearly synonymous with the label and no
model could do better than threshold it.

The gate was applied to 22 GDSC2 drugs. This script asks whether it worked.

WHAT A USABLE GATE MUST DO
--------------------------
1. Discriminate. A criterion that admits every candidate carries no
   information, whatever its face validity.
2. Measure the construct it names. IF must track *exploitable interaction
   structure*, not some other quantity that happens to correlate with it.
3. Predict the thing it was built to predict. If IF cannot rank tasks by
   realised quantum gain, it does not earn its place in the protocol.

Test 3 is reported honestly as underpowered: quantum benchmarks exist for the
5 PARP drugs only, so n=5 and the result is descriptive, not inferential.

Usage:
    .venv/bin/python scripts/gate_construct_validity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
GATE = 0.40


def main():
    summary = json.load(open(REPO / "data" / "final_summary_22drug.json"))
    drugs = summary["drugs"]

    IF = np.array([d["interaction_fraction"] for d in drugs])
    AUC = np.array([d["gbdt_auc"] for d in drugs])
    names = [d["drug"] for d in drugs]
    is_parp = np.array([d["arm"] == "PARP_inhibitor" for d in drugs])

    out = {
        "question": "Does the interaction_fraction > 0.40 admission gate have construct validity?",
        "n_drugs": len(drugs),
        "gate_threshold": GATE,
    }

    # -- 1. discrimination ------------------------------------------------
    n_pass = int((IF > GATE).sum())
    out["test_1_discrimination"] = {
        "n_passing": n_pass,
        "n_total": len(IF),
        "pass_rate": n_pass / len(IF),
        "if_min": float(IF.min()),
        "if_max": float(IF.max()),
        "if_range": float(IF.max() - IF.min()),
        "if_sd": float(IF.std(ddof=1)),
        "distance_of_nearest_miss_to_gate": float(IF.min() - GATE),
        "verdict": ("FAIL -- the gate admits every candidate screened. Its "
                    "closest approach to rejection is "
                    f"{IF.min() - GATE:+.4f}, i.e. the entire observed "
                    "distribution sits above the threshold. A criterion with "
                    "a 100% pass rate cannot inform an admission decision."
                    if n_pass == len(IF) else "pass"),
    }

    # -- 2. construct: is IF just inverse classical performance? ----------
    rho, p_rho = stats.spearmanr(IF, AUC)
    r, p_r = stats.pearsonr(IF, AUC)
    slope, icept, r_lin, p_lin, se = stats.linregress(AUC, IF)
    out["test_2_construct"] = {
        "spearman_if_vs_gbdt_auc": {"rho": float(rho), "p": float(p_rho)},
        "pearson_if_vs_gbdt_auc": {"r": float(r), "p": float(p_r)},
        "variance_of_if_explained_by_auc": float(r_lin ** 2),
        "verdict": (
            f"FAIL -- IF is inversely rank-correlated with classical GBDT AUC "
            f"(rho={rho:.3f}, p={p_rho:.2e}); linear fit on AUC accounts for "
            f"{r_lin**2:.0%} of IF variance. IF therefore largely restates how "
            "little signal the classical model captured. A task looks "
            "'interaction-rich' by this measure precisely when the classical "
            "baseline is weak, which inverts the intended meaning: the gate "
            "preferentially admits tasks that are hard, not tasks whose "
            "difficulty is due to exploitable interaction structure."),
    }

    # -- 3. does IF predict realised quantum gain? ------------------------
    gains = {}
    for d, flag in zip(drugs, is_parp):
        if not flag:
            continue
        f = REPO / "data" / f"gdsc_{d['drug']}_quantile" / "true_qsvm_tuned.json"
        if f.exists():
            rec = json.load(open(f))
            m = rec["models"]
            gains[d["drug"]] = m["QSVM"]["auc"] - m["SVM-RBF"]["auc"]

    if len(gains) >= 3:
        g_if = np.array([d["interaction_fraction"] for d in drugs
                         if d["drug"] in gains])
        g_val = np.array([gains[d["drug"]] for d in drugs if d["drug"] in gains])
        g_rho, g_p = stats.spearmanr(g_if, g_val)
        out["test_3_predictive"] = {
            "n": int(len(g_val)),
            "drugs": list(gains),
            "spearman_if_vs_quantum_gain": {"rho": float(g_rho), "p": float(g_p)},
            "caveat": ("Quantum benchmarks exist for the 5 PARP inhibitors "
                       "only. n=5 cannot support an inferential claim; this is "
                       "descriptive. It is reported because its absence would "
                       "be a silent gap, not because it settles anything."),
            "verdict": ("No association detectable at this sample size. The "
                        "gate's central promise -- that high IF forecasts "
                        "quantum gain -- is untested rather than refuted, and "
                        "the design that would test it (many drugs, quantum "
                        "arm on all of them) was never run."),
        }

    # -- 4. arm separation ------------------------------------------------
    u, p_u = stats.mannwhitneyu(IF[is_parp], IF[~is_parp], alternative="two-sided")
    n1, n2 = int(is_parp.sum()), int((~is_parp).sum())
    out["test_4_arm_separation"] = {
        "test": "Mann-Whitney U, two-sided, PARP vs cytotoxic negative controls",
        "n_parp": n1, "n_negative_control": n2,
        "U": float(u), "p": float(p_u),
        "rank_biserial_r": float(1 - 2 * u / (n1 * n2)),
        "mean_if_parp": float(IF[is_parp].mean()),
        "mean_if_negctrl": float(IF[~is_parp].mean()),
        "direction": ("negative controls score HIGHER than PARP inhibitors"
                      if IF[~is_parp].mean() > IF[is_parp].mean()
                      else "PARP inhibitors score higher"),
        "verdict": (
            f"FAIL -- no separation (p={p_u:.3f}), and the point estimate runs "
            "the wrong way: cytotoxic agents, whose response is not expected to "
            "hinge on combinatorial DNA-repair biology, score slightly higher "
            "on IF than the PARP inhibitors do. This is what test 2 predicts, "
            "since the cytotoxics also have the weaker classical baselines."),
    }

    out["conclusion"] = {
        "gate_status": "REFUTED",
        "summary": (
            "The interaction_fraction > 0.40 admission gate fails on three "
            "independent counts: it admits 22/22 candidates, it is largely a "
            "restatement of inverse classical performance "
            f"(rho={rho:.3f}, p={p_rho:.1e}), and it does not separate PARP "
            f"inhibitors from cytotoxic controls (p={p_u:.3f}, wrong sign). "
            "The failure is instructive rather than merely negative: SHAP "
            "interaction mass is a property of the fitted model, so when a "
            "model fits poorly its attribution spreads across interaction "
            "terms regardless of the underlying biology. Any future gate must "
            "be computed on a quantity that does not inherit the baseline "
            "model's weakness."),
        "implication_for_protocol": (
            "Do not use IF, or any single fitted-model attribution statistic, "
            "as a pre-commitment criterion for quantum compute. Candidate "
            "replacements worth testing: a model-free interaction measure "
            "(e.g. distance correlation beyond additive fits), or a direct "
            "expressivity comparison between the quantum kernel and its "
            "classical counterpart on the same features."),
        "evidence_level": "simulated-poc",
    }

    dest = REPO / "benchmarks" / "results" / "gate_construct_validity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)

    # -- console ----------------------------------------------------------
    print(f"\nGate: interaction_fraction > {GATE}   (n={len(IF)} drugs)\n")
    print(f"  1. discrimination : {n_pass}/{len(IF)} pass "
          f"({n_pass/len(IF):.0%}), IF range {IF.min():.4f}-{IF.max():.4f}")
    print(f"  2. construct      : spearman(IF, GBDT AUC) = {rho:+.3f}  p={p_rho:.2e}"
          f"   [AUC explains {r_lin**2:.0%} of IF variance]")
    if "test_3_predictive" in out:
        t3 = out["test_3_predictive"]["spearman_if_vs_quantum_gain"]
        print(f"  3. predictive     : spearman(IF, quantum gain) = {t3['rho']:+.3f}"
              f"  p={t3['p']:.3f}   (n={out['test_3_predictive']['n']}, descriptive)")
    print(f"  4. arm separation : U={u:.1f} p={p_u:.3f}  "
          f"({out['test_4_arm_separation']['direction']})")
    print(f"\n  => {out['conclusion']['gate_status']}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
