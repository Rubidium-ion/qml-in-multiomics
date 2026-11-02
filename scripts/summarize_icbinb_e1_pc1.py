#!/usr/bin/env python3
"""Summarize checkpointed E1 per-drug leakage-safe PC1 results."""
import csv, json
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "results"
RUNS = OUT / "icbinb_e1_pc1_runs"

def association(x, y):
    r = spearmanr(x, y)
    return {"spearman_rho": float(r.statistic), "p": float(r.pvalue)}

def main():
    manifest = json.load(open(OUT / "icbinb_d3_frozen_selection_20drug.json"))
    c1 = {x["drug"]: x for x in json.load(open(OUT / "icbinb_c1_shap_convention_IF.json"))["cohorts"]}
    tuned = {x["drug"]: x for x in csv.DictReader(open(OUT / "icbinb_d3_frozen_pi16_3seed_tuned_classical_per_drug.csv"))}
    rows = []
    for selected in manifest["selected"]:
        drug = selected["drug"]
        checkpoint = RUNS / f"{drug}.json"
        if not checkpoint.exists():
            raise RuntimeError(f"missing checkpoint: {checkpoint}")
        row = json.load(open(checkpoint))
        row.update({
            "class": selected["class"],
            "if_full": float(c1[drug]["if_full"]),
            "if_gate_full_040": float(c1[drug]["if_full"]) >= 0.40,
            "qsvm_auc": float(tuned[drug]["qsvm_auc"]),
            "tuned_gbdt_auc": float(tuned[drug]["tuned_gbdt_auc"]),
            "qsvm_minus_tuned_gbdt": float(tuned[drug]["delta_qsvm_minus_tuned_gbdt_auc"]),
        })
        rows.append(row)
    fields = ["drug", "class", "n", "pc1_auc", "pc1_auc_seed_sd", "pc1_explained_variance_train", "pc1_test_projection_variance", "pc1_response_spearman_train", "pc1_flag_perm95", "pc1_perm_null_95", "rank_gaussian_pc1_auc", "random_orthogonal_auc", "if_full", "if_gate_full_040", "qsvm_auc", "tuned_gbdt_auc", "qsvm_minus_tuned_gbdt"]
    with open(OUT / "icbinb_e1_pc1_per_drug.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])
    pc1_ev = [r["pc1_explained_variance_train"] for r in rows]
    flags = {
        "if_gate_full_040": sum(r["if_gate_full_040"] for r in rows),
        "pc1_perm95": sum(r["pc1_flag_perm95"] for r in rows),
        "intersection": sum(r["if_gate_full_040"] and r["pc1_flag_perm95"] for r in rows),
        "union": sum(r["if_gate_full_040"] or r["pc1_flag_perm95"] for r in rows),
    }
    strata = {}
    for cls in ("PARP", "cytotoxic"):
        subset = [r for r in rows if r["class"] == cls]
        strata[cls] = {
            "n": len(subset),
            "mean_pc1_auc": sum(r["pc1_auc"] for r in subset) / len(subset),
            "mean_qsvm_minus_tuned_gbdt": sum(r["qsvm_minus_tuned_gbdt"] for r in subset) / len(subset),
        }
    summary = {
        "format": "icbinb_e1_leakage_safe_pc1.v1",
        "n_drugs": len(rows),
        "seeds": [42, 314, 2718],
        "protocol": "For each drug/seed/outer fold, preprocessing and PCA fit only on outer-train rows; held-out rows projected using the training fit. PC1-only logistic baseline evaluated on held-out fold. QK and tuned GBDT reused from frozen D3; no QK rerun.",
        "pc1_flag": "PC1 train explained variance exceeds 95th percentile of feature-wise independently permuted expression null (three-seed aggregate per drug).",
        "per_drug_csv": "benchmarks/results/icbinb_e1_pc1_per_drug.csv",
        "flags": flags,
        "associations": {
            "pc1ev_vs_if_full": association(pc1_ev, [r["if_full"] for r in rows]),
            "pc1ev_vs_tuned_gbdt_auc": association(pc1_ev, [r["tuned_gbdt_auc"] for r in rows]),
            "pc1ev_vs_qsvm_minus_tuned_gbdt": association(pc1_ev, [r["qsvm_minus_tuned_gbdt"] for r in rows]),
        },
        "strata": strata,
        "negative_controls": "Per-drug rank/Gaussianized PC1 and random orthogonal 1D projections are recorded; all fitted in outer training folds. PC2-PC10 and tissue-within centering are not separately run in this compact execution; sparse tissue strata require separately pre-specified handling.",
    }
    json.dump(summary, open(OUT / "icbinb_e1_pc1_summary.json", "w"), indent=2)
    print(json.dumps({"flags": flags, "associations": summary["associations"]}, indent=2))

if __name__ == "__main__":
    main()
