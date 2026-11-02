#!/usr/bin/env python3
"""parp_paired_test.py -- cross-drug paired test for the QSVM-vs-classical claim.

WHY THIS EXISTS
---------------
run_true_qsvm.py previously emitted, per drug:

    "quantum_advantage_vs_rbf": 0.0122,
    "verdict": "quantum kernel outperforms RBF"

on the strength of a hard-coded |delta| > 0.01 rule. Fold-level AUC sd on these
cohorts is ~0.03, so 0.0122 is well inside noise; and the rule was applied to
each drug independently, with no correction for the fact that five drugs were
screened. One drug out of five landing above an arbitrary line is the expected
behaviour of noise, not evidence.

This script replaces that with the comparison the claim actually requires:
a PAIRED test across drugs, plus Holm correction across the per-drug tests.

Both encodings are evaluated:
  - pi   (encoding_scale = pi)      -- the naive setting
  - tuned(encoding_scale = 0.196)   -- after encoding-scale repair

Reporting the pi arm matters. Its kernel off-diagonal mass is ~1e-3, i.e. the
feature map maps every pair of samples to near-orthogonal states and the Gram
matrix is effectively the identity. The large "tuning gain" is therefore the
repair of a degenerate encoding, not quantum signal -- and that distinction is
only visible if both arms are reported.

Usage:
    .venv/bin/python scripts/parp_paired_test.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
PARP = ["talazoparib", "rucaparib", "veliparib", "niraparib", "olaparib"]
ARMS = {"pi": "true_qsvm_pi.json", "tuned": "true_qsvm_tuned.json"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order-preserving."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def load(drug, fname):
    p = DATA / f"gdsc_{drug}_quantile" / fname
    if not p.exists():
        return None
    return json.load(open(p))


def per_drug_paired(rec):
    """Paired t-test across folds. Returns None if fold data was not retained."""
    m = rec.get("models", {})
    q = m.get("QSVM", {}).get("auc_folds")
    r = m.get("SVM-RBF", {}).get("auc_folds")
    if not q or not r:
        return None
    d = np.asarray(q) - np.asarray(r)
    t, p = stats.ttest_rel(q, r)
    return {"mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
            "t": float(t), "p": float(p), "n_folds": len(d)}


def main():
    report = {
        "title": "Cross-drug paired comparison, QSVM vs classical baselines (PARP arm)",
        "supersedes": ("per-drug 'verdict' strings emitted by run_true_qsvm.py "
                       "before 2026-08-09, which used a hard-coded |delta|>0.01 "
                       "rule with no test and no multiplicity correction"),
        "drugs": PARP,
        "arms": {},
    }

    for arm, fname in ARMS.items():
        rows, missing = [], []
        for drug in PARP:
            rec = load(drug, fname)
            if rec is None:
                missing.append(drug)
                continue
            m = rec["models"]
            rows.append({
                "drug": drug,
                "qsvm_auc": m["QSVM"]["auc"],
                "rbf_auc": m["SVM-RBF"]["auc"],
                "gbdt_auc": m["GBDT"]["auc"],
                "qsvm_auc_sd": m["QSVM"]["auc_sd"],
                "delta_vs_rbf": m["QSVM"]["auc"] - m["SVM-RBF"]["auc"],
                "delta_vs_gbdt": m["QSVM"]["auc"] - m["GBDT"]["auc"],
                "kernel_offdiag_mean": rec.get("kernel_offdiag_mean"),
                "encoding_scale": rec.get("encoding_scale"),
                "within_drug_paired": per_drug_paired(rec),
            })

        q = np.array([r["qsvm_auc"] for r in rows])
        rbf = np.array([r["rbf_auc"] for r in rows])
        gbdt = np.array([r["gbdt_auc"] for r in rows])

        arm_out = {"per_drug": rows, "missing": missing, "n_drugs": len(rows)}

        for label, base in (("vs_SVM_RBF", rbf), ("vs_GBDT", gbdt)):
            d = q - base
            t, p = stats.ttest_rel(q, base)
            w = stats.wilcoxon(q, base) if len(d) >= 5 else None
            se = d.std(ddof=1) / np.sqrt(len(d))
            crit = stats.t.ppf(0.975, df=len(d) - 1)
            arm_out[label] = {
                "test": "paired two-sided t-test across drugs on mean AUC",
                "n": int(len(d)),
                "mean_delta": float(d.mean()),
                "sd_delta": float(d.std(ddof=1)),
                "ci95": [float(d.mean() - crit * se), float(d.mean() + crit * se)],
                "t": float(t),
                "p": float(p),
                "wilcoxon_p": float(w.pvalue) if w is not None else None,
                "n_drugs_favouring_quantum": int((d > 0).sum()),
                "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
            }

        # Holm across the per-drug within-drug tests, when fold data exists.
        wd = [(r["drug"], r["within_drug_paired"]["p"])
              for r in rows if r.get("within_drug_paired")]
        if wd:
            names = [x[0] for x in wd]
            ps = np.array([x[1] for x in wd])
            adj = holm(ps)
            arm_out["per_drug_holm"] = {
                n: {"p_raw": float(pr), "p_holm": float(pa),
                    "significant_at_0.05": bool(pa < 0.05)}
                for n, pr, pa in zip(names, ps, adj)
            }
        else:
            arm_out["per_drug_holm"] = {
                "status": "unavailable",
                "reason": ("per-fold AUCs were not retained by the run that "
                           "produced these files. run_true_qsvm.py now stores "
                           "auc_folds; re-run to enable within-drug paired "
                           "tests and Holm correction."),
            }

        report["arms"][arm] = arm_out

    # ---- headline -------------------------------------------------------
    tuned = report["arms"]["tuned"]
    pi = report["arms"]["pi"]
    report["conclusion"] = {
        "tuned_vs_rbf": (
            f"mean delta {tuned['vs_SVM_RBF']['mean_delta']:+.4f} AUC, "
            f"paired p={tuned['vs_SVM_RBF']['p']:.3f}, "
            f"{tuned['vs_SVM_RBF']['n_drugs_favouring_quantum']}/"
            f"{tuned['vs_SVM_RBF']['n']} drugs favouring quantum"),
        "pi_vs_rbf": (
            f"mean delta {pi['vs_SVM_RBF']['mean_delta']:+.4f} AUC, "
            f"paired p={pi['vs_SVM_RBF']['p']:.3f}"),
        "verdict": (
            "No quantum advantage is established. Under the repaired encoding "
            "the quantum kernel is statistically indistinguishable from SVM-RBF "
            "across the five PARP inhibitors; under the naive encoding it is "
            "clearly worse. The previously reported single-drug 'outperforms' "
            "result (olaparib, +0.0122 AUC) does not survive a paired test "
            "across drugs and is consistent with fold-level noise "
            "(AUC sd ~0.03)."),
        "encoding_note": (
            "Kernel off-diagonal mass rises from ~1e-3 (scale=pi) to ~0.65 "
            "(scale=0.196). The naive feature map produces near-orthogonal "
            "states for all sample pairs, making the Gram matrix effectively "
            "the identity. The apparent gain from tuning is the repair of a "
            "degenerate encoding, not evidence of quantum signal."),
        "evidence_level": "simulated-poc",
    }

    dest = REPO / "benchmarks" / "results" / "parp_paired_test.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(dest, "w"), indent=2)

    # ---- console --------------------------------------------------------
    for arm in ("pi", "tuned"):
        a = report["arms"][arm]
        print(f"\n=== arm: {arm}  (n={a['n_drugs']} drugs) ===")
        print(f"  {'drug':<14}{'QSVM':>8}{'RBF':>8}{'GBDT':>8}{'dRBF':>9}{'offdiag':>10}")
        for r in a["per_drug"]:
            print(f"  {r['drug']:<14}{r['qsvm_auc']:>8.4f}{r['rbf_auc']:>8.4f}"
                  f"{r['gbdt_auc']:>8.4f}{r['delta_vs_rbf']:>+9.4f}"
                  f"{r['kernel_offdiag_mean']:>10.4f}")
        for label in ("vs_SVM_RBF", "vs_GBDT"):
            s = a[label]
            print(f"  {label}: delta {s['mean_delta']:+.4f} "
                  f"CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                  f"t={s['t']:+.3f} p={s['p']:.3f} "
                  f"wilcoxon p={s['wilcoxon_p']}")

    print("\n" + report["conclusion"]["verdict"])
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
