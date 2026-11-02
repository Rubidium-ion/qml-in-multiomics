#!/usr/bin/env python3
"""Assert every quantitative claim in docs/abstract.md against its source JSON.

An abstract is the part reviewers read and the part most likely to drift from the
analysis as numbers are re-run. This makes that drift a build failure.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(rel):
    with open(REPO / rel) as fh:
        return json.load(fh)


def main():
    text = (REPO / "docs" / "abstract.md").read_text()
    gate = load("benchmarks/results/gate_construct_validity.json")
    paired = load("benchmarks/results/parp_paired_test.json")
    summary = load("data/final_summary_22drug.json")
    tuned = load("data/gdsc_olaparib_quantile/true_qsvm_tuned.json")

    checks = []

    def ck(cond, label, shown, actual):
        checks.append((bool(cond), label, shown, actual))

    # --- cohort ---
    n = [d["n_samples"] for d in summary["drugs"]]
    arms = [d["arm"] for d in summary["drugs"]]
    ck(len(summary["drugs"]) == 22 and "22 GDSC2" in text, "22 cohorts", "22", len(summary["drugs"]))
    ck(arms.count("PARP_inhibitor") == 5 and "5 PARP" in text, "5 PARP", "5", arms.count("PARP_inhibitor"))
    ck(arms.count("negative_control") == 17 and "17 cytotoxic" in text,
       "17 controls", "17", arms.count("negative_control"))
    ck(min(n) == 715 and max(n) == 962 and "715–962" in text, "n range", "715-962", f"{min(n)}-{max(n)}")

    # --- gate ---
    t1, t2, t4 = gate["test_1_discrimination"], gate["test_2_construct"], gate["test_4_arm_separation"]
    ck(gate["gate_threshold"] == 0.40 and "0.40" in text, "gate threshold", "0.40", gate["gate_threshold"])
    ck(t1["n_passing"] == 22 and t1["n_total"] == 22 and "22/22" in text,
       "admitted all", "22/22", f"{t1['n_passing']}/{t1['n_total']}")
    rho = t2["spearman_if_vs_gbdt_auc"]["rho"]
    ck(round(rho, 2) == -0.85 and "−0.85" in text, "construct rho", "-0.85", round(rho, 4))
    p2 = t2["spearman_if_vs_gbdt_auc"]["p"]
    ck(abs(p2 - 6.9e-7) < 5e-8 and "6.9 × 10⁻⁷" in text, "construct p", "6.9e-7", f"{p2:.2e}")
    r2 = t2.get("variance_of_if_explained_by_auc")
    ck(r2 is not None and round(r2 * 100) == 67 and "67%" in text,
       "variance explained", "67%", f"{r2:.4f}" if r2 is not None else None)
    ck("HIGHER" in t4["direction"].upper() and "higher than PARP" in text,
       "controls score higher", "controls higher", t4["direction"])

    # --- encoding / kernel ---
    pi_arm, tuned_arm = paired["arms"]["pi"], paired["arms"]["tuned"]
    offdiag = sum(r["kernel_offdiag_mean"] for r in pi_arm["per_drug"]) / 5
    ck(abs(offdiag - 9.66e-4) < 1e-5 and "9.7 × 10⁻⁴" in text,
       "naive off-diagonal", "9.7e-4", f"{offdiag:.3e}")
    gain = tuned_arm["vs_SVM_RBF"]["mean_delta"] - pi_arm["vs_SVM_RBF"]["mean_delta"]
    ck(abs(gain - 0.058) < 0.002 and "0.058" in text, "rescaling gain", "0.058", round(gain, 4))
    ck(abs(tuned["encoding_scale"] - 3.14159265 / 16) < 1e-4 and "π/16" in text,
       "tuned scale", "pi/16", tuned["encoding_scale"])
    ck(tuned["n_qubits"] == 12 and "12-qubit" in text, "qubits", "12", tuned["n_qubits"])

    # --- headline comparison ---
    d_rbf = tuned_arm["vs_SVM_RBF"]
    ck(round(d_rbf["mean_delta"], 3) == -0.003 and "−0.003" in text,
       "tuned vs RBF delta", "-0.003", round(d_rbf["mean_delta"], 4))
    ck(round(d_rbf["p"], 2) == 0.61 and "0.61" in text, "tuned vs RBF p", "0.61", round(d_rbf["p"], 4))
    n_sig = sum(1 for v in pi_arm["per_drug_holm"].values() if v["significant_at_0.05"])
    ck(n_sig == 3 and "(3/5)" in text, "naive Holm-significant", "3/5", f"{n_sig}/5")
    sig_drugs = [k for k, v in pi_arm["per_drug_holm"].items() if v["significant_at_0.05"]]
    delta_by_drug = {r["drug"]: r["delta_vs_rbf"] for r in pi_arm["per_drug"]}
    favours_classical = bool(sig_drugs) and all(delta_by_drug[k] < 0 for k in sig_drugs)
    ck(favours_classical and "favoured the" in text and "classical kernel" in text,
       "all significant favour classical", "all classical",
       {k: round(delta_by_drug[k], 4) for k in sig_drugs})

    # --- claims that must NOT appear in the abstract prose ---
    # Scope to the abstract bodies only: the trailing "Notes for assembly" section
    # deliberately discusses the claims we chose to exclude, and must not trip this.
    body = text.split("## Notes for assembly")[0]
    for banned in ("0.0177", "quantum advantage is established", "outperform"):
        ck(banned not in body, f"absent from abstract: {banned!r}",
           "absent", "absent" if banned not in body else "PRESENT")
    # ...and the rationale for excluding the GBDT delta must still be on record.
    ck("0.0177" in text, "exclusion of +0.0177 documented in notes", "documented",
       "documented" if "0.0177" in text else "MISSING")

    width = max(len(c[1]) for c in checks)
    bad = 0
    for ok, label, shown, actual in checks:
        if not ok:
            bad += 1
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:<{width}}  shown={shown!s:<12} source={actual}")

    print()
    if bad:
        print(f"{bad}/{len(checks)} ABSTRACT CLAIMS DO NOT MATCH SOURCE")
        return 1
    print(f"ALL {len(checks)} ABSTRACT CLAIMS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
