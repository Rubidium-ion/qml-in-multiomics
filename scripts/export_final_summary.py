#!/usr/bin/env python3
"""export_final_summary.py -- terminal aggregation for the QML/PARP study.

AGGREGATION ONLY. This script reads prescreen.json files and the existing QML
bench table. It trains nothing and spends no new compute, honouring the agreed
stop line.

SCOPE NOTES THAT MATTER FOR THE PAPER
-------------------------------------
1. 22 drugs are screened on disk. The "locked list" circulated as ~19 drugs is
   internally inconsistent (it says 14 negative controls but enumerates 16, and
   omits Vinblastine, which was screened under the same protocol). We export
   ALL 22 and mark membership explicitly rather than silently dropping a drug
   that was already measured. Dropping a screened, protocol-conformant drug
   because it is absent from a list written after the fact is post-hoc
   exclusion, which is the failure mode the no-cherry-picking rule exists to
   prevent.

2. Quantum bench results (q_tuned / delta_tuned_vs_rbf) exist for the FIVE PARP
   drugs only. Any correlation between interaction_fraction and "quantum gain"
   is therefore computed on n=5 and is reported with that caveat attached. It
   is not evidence at that sample size; it is a descriptive number.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARP = {"olaparib", "rucaparib", "niraparib", "veliparib", "talazoparib"}

MECHANISM = {
    "paclitaxel": "microtubule", "docetaxel": "microtubule",
    "vinorelbine": "microtubule", "vinblastine": "microtubule",
    "5fluorouracil": "antimetabolite", "gemcitabine": "antimetabolite",
    "methotrexate": "antimetabolite", "cytarabine": "antimetabolite",
    "cisplatin": "platinum/alkylating", "oxaliplatin": "platinum/alkylating",
    "temozolomide": "alkylating",
    "teniposide": "topoisomerase-II", "irinotecan": "topoisomerase-I",
    "camptothecin": "topoisomerase-I",
    "bleomycin": "DNA strand break",
    "bortezomib": "proteasome", "palbociclib": "CDK4/6",
}

# Drugs enumerated on Bio's locked list (5 PARPi + the 16 it actually names).
LOCKED = PARP | {
    "paclitaxel", "docetaxel", "vinorelbine",
    "5fluorouracil", "gemcitabine", "methotrexate", "cytarabine",
    "cisplatin", "oxaliplatin", "temozolomide",
    "teniposide", "irinotecan", "camptothecin",
    "bleomycin", "bortezomib", "palbociclib",
}

NOTES = {
    "bleomycin": "DNA strand-break inducer; IF may be partly driven by DNA "
                 "repair pathway expression, so its mechanism is NOT specific "
                 "to the negative-control premise (flagged by bio).",
    "teniposide": "substituted for Etoposide, absent from GDSC2 (same TOP2 "
                  "mechanism).",
    "vinorelbine": "substituted for Vincristine, absent from GDSC2.",
    "palbociclib": "boundary case: same indication (breast), different "
                   "mechanism (CDK4/6).",
    "vinblastine": "screened under the identical protocol but omitted from the "
                   "later locked list; retained to avoid post-hoc exclusion.",
}

UNAVAILABLE = {
    "Vincristine": "not in GDSC2 (286 drugs enumerated; no alias)",
    "Carboplatin": "not in GDSC2; no substitute (accepted by bio)",
    "Etoposide": "not in GDSC2; Teniposide substituted (same TOP2 mechanism)",
}


def main():
    bench = {r["drug"]: r for r in json.load(
        open(os.path.join(REPO, "data", "option_b_master_table.json")))}

    drugs = []
    for p in sorted(glob.glob(os.path.join(REPO, "data", "gdsc_*_quantile",
                                           "prescreen.json"))):
        j = json.load(open(p))
        name = os.path.basename(os.path.dirname(p))[len("gdsc_"):-len("_quantile")]
        b = bench.get(name)
        rec = {
            "drug": name,
            "arm": "PARP_inhibitor" if name in PARP else "negative_control",
            "mechanism": "PARP inhibitor" if name in PARP
                         else MECHANISM.get(name, "unclassified"),
            "on_locked_list": name in LOCKED,
            "interaction_fraction": j["interaction_fraction"],
            "gbdt_auc": j["gbdt"]["auc"],
            "gbdt_auc_sd": j["gbdt"]["auc_sd"],
            "balanced_accuracy": j["gbdt"]["balanced_accuracy"],
            "n_samples": j["n_samples"],
            "majority_baseline": j["majority_baseline"],
            "passes_if_gate_0.40": j["interaction_fraction"] > 0.40,
            "top3_interaction_pairs": [
                {"pair": f"{d['a']}x{d['b']}", "strength": round(d["strength"], 6)}
                for d in j["top_pairs"][:3]],
            "top3_main_effects": [
                {"feature": d["feature"], "strength": round(d["strength"], 6)}
                for d in j["top_main_effects"][:3]],
            "brca_ablation_auc": j["ablation_brca"]["auc"],
            "brca_ablation_if": j["ablation_brca"]["interaction_fraction"],
            "note": NOTES.get(name),
        }
        if b:
            rec["qml_bench"] = {
                "svm_rbf_auc": b["svm_rbf"],
                "q_tuned_auc": b["q_tuned"],
                "delta_tuned_vs_rbf": b["delta_tuned_vs_rbf"],
                "tuning_gain": b["tuning_gain"],
            }
        drugs.append(rec)

    pv = np.array([d["interaction_fraction"] for d in drugs
                   if d["arm"] == "PARP_inhibitor"])
    nv = np.array([d["interaction_fraction"] for d in drugs
                   if d["arm"] == "negative_control"])

    U2, p2 = stats.mannwhitneyu(nv, pv, alternative="two-sided")
    rb = 2.0 * U2 / (len(nv) * len(pv)) - 1.0

    # Same test restricted to the locked list, so reviewers can see the choice
    # of drug set does not rescue the result.
    pl = np.array([d["interaction_fraction"] for d in drugs
                   if d["arm"] == "PARP_inhibitor" and d["on_locked_list"]])
    nl = np.array([d["interaction_fraction"] for d in drugs
                   if d["arm"] == "negative_control" and d["on_locked_list"]])
    U2l, p2l = stats.mannwhitneyu(nl, pl, alternative="two-sided")

    all_if = np.array([d["interaction_fraction"] for d in drugs])
    all_auc = np.array([d["gbdt_auc"] for d in drugs])
    rho_auc, p_auc = stats.spearmanr(all_if, all_auc)

    # IF vs quantum gain -- PARP arm only, n=5.
    qd = [d for d in drugs if "qml_bench" in d]
    q_if = np.array([d["interaction_fraction"] for d in qd])
    q_dl = np.array([d["qml_bench"]["delta_tuned_vs_rbf"] for d in qd])
    rho_q, p_q = stats.spearmanr(q_if, q_dl)

    print(f"drugs exported            : {len(drugs)} "
          f"({len(pv)} PARPi / {len(nv)} negative control)")
    print(f"on locked list            : {sum(d['on_locked_list'] for d in drugs)}")
    print()
    print(f"PARP  IF mean={pv.mean():.4f} median={np.median(pv):.4f}")
    print(f"NEG   IF mean={nv.mean():.4f} median={np.median(nv):.4f}")
    print(f"Mann-Whitney U (all 22)   U={U2:.1f} p={p2:.4f} r_rb={rb:+.4f}")
    print(f"Mann-Whitney U (locked 21) U={U2l:.1f} p={p2l:.4f}")
    print()
    print(f"IF gate pass rate         : {(all_if>0.40).sum()}/{len(all_if)}")
    print(f"Spearman(IF, classic AUC) : rho={rho_auc:+.3f} p={p_auc:.2e}")
    print(f"Spearman(IF, quantum dlt) : rho={rho_q:+.3f} p={p_q:.4f}  (n={len(qd)})")
    print()

    out = {
        "title": "Final summary -- PARP/GDSC2 QML admission study",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "compute_note": "aggregation only; no models trained by this script",
        "n_drugs": len(drugs),
        "arms": {"PARP_inhibitor": int(len(pv)),
                 "negative_control": int(len(nv))},
        "drug_set_note":
            "22 drugs screened under one protocol (per-drug quantile:25). The "
            "circulated locked list is internally inconsistent (states 14 "
            "negative controls, enumerates 16) and omits Vinblastine; all 22 "
            "are exported with on_locked_list flags rather than dropping a "
            "measured drug post hoc.",
        "primary_result": {
            "test": "Mann-Whitney U, two-sided, on interaction_fraction",
            "all_22": {"U": float(U2), "p": float(p2),
                       "rank_biserial_r": float(rb)},
            "locked_subset_21": {"U": float(U2l), "p": float(p2l),
                                 "n_parp": int(len(pl)), "n_neg": int(len(nl))},
            "direction": "negative controls score HIGHER IF than PARP inhibitors",
            "target_p_lt_0_01_met": bool(p2 < 0.01),
            "interpretation":
                "The IF criterion does not separate PARP inhibitors from "
                "cytotoxic negative controls. The earlier p=0.056 did not "
                "survive the increase in sample size, which is the expected "
                "behaviour of a small-sample fluctuation rather than of a "
                "diluted true effect.",
        },
        "gate_construct_validity": {
            "pass_rate": f"{int((all_if>0.40).sum())}/{len(all_if)}",
            "if_range": [float(all_if.min()), float(all_if.max())],
            "if_spread": float(all_if.max() - all_if.min()),
            "spearman_if_vs_classical_auc": {"rho": float(rho_auc),
                                             "p": float(p_auc)},
            "verdict":
                "Every screened drug clears the 0.40 gate, and IF is strongly "
                "inversely rank-correlated with classical GBDT AUC. IF "
                "therefore largely restates how little signal the classical "
                "model captured, rather than identifying exploitable "
                "interaction structure. The admission criterion fails "
                "construct validity independently of the arm comparison.",
        },
        "if_vs_quantum_gain": {
            "spearman_rho": float(rho_q),
            "p": float(p_q),
            "n": int(len(qd)),
            "caveat":
                "Computed on the 5 PARP drugs only, the sole drugs with QML "
                "bench results. n=5 is not sufficient to support an inferential "
                "claim; this is descriptive.",
        },
        "drugs_requested_but_unavailable": UNAVAILABLE,
        "drugs": sorted(drugs, key=lambda d: (d["arm"],
                                              -d["interaction_fraction"])),
    }
    dest = os.path.join(REPO, "data", "final_summary_22drug.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
