#!/usr/bin/env python
"""Freeze the D3 confirmatory 20-drug quantum benchmark selection.

Bio design (see shared/bio/Bio_statistical_revision_addendum.md D3):
- Sampling frame = 22 eligible GDSC2 cohorts.
- Stratified: 4 drugs from each IF quintile, balanced across classical-performance
  tertiles where feasible.
- Existing 5 PARP cohorts are the OBSERVED PILOT (already run, reps=1, fixed scale,
  single seed). The newly selected >=15 drugs form the CONFIRMATORY EXTENSION.
- CRITICAL: selection is frozen and committed BEFORE any new quantum result is run.
  No drug may be chosen after viewing quantum performance.
- Deterministic replacement rule must be logged.

Selection inputs are ALL pre-quantum: IF (full convention, shap 0.52.0) and
classical GBDT AUC — both available in prescreen.json / icbinb_c1 table.

This script writes the frozen selection manifest + timestamp; run it FIRST and
commit before launching any new QSVM job.
"""
import json, csv, hashlib, os, sys, time, uuid
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

SEED = 42
GATE = 0.40
PILOT_PARP = ["olaparib", "niraparib", "rucaparib", "talazoparib", "veliparib"]
N_PER_QUINTILE = 4
N_TOTAL = 20  # 4 drugs x 5 quintiles

# Load sampling frame (IF full + gbdt_auc)
c1 = json.load(open("benchmarks/results/icbinb_c1_shap_convention_IF.json"))
frame = []
for c in c1["cohorts"]:
    drug = c["drug"]
    ps = json.load(open(f"data/gdsc_{drug}_quantile/prescreen.json"))
    frame.append({
        "drug": drug,
        "class": "PARP" if drug in PILOT_PARP else "cytotoxic",
        "is_pilot": drug in PILOT_PARP,
        "if_full": c["if_full"],
        "gbdt_auc": ps.get("gbdt", {}).get("auc"),
        "n": c["n"],
    })
frame.sort(key=lambda x: x["drug"])

# IF quintile edges
ifs = np.array([r["if_full"] for r in frame])
edges = np.quantile(ifs, [0, .2, .4, .6, .8, 1.0])
for r in frame:
    q = 0
    for qi in range(5):
        if edges[qi] <= r["if_full"] < edges[qi+1] or (qi == 4 and r["if_full"] == edges[-1]):
            q = qi
            break
    r["quintile"] = q

# Classical AUC tertiles (over the 22 frame, for balance)
aucs = np.array([r["gbdt_auc"] for r in frame])
aterts = np.quantile(aucs, [0, 1/3, 2/3, 1.0])
for r in frame:
    t = 0
    for ti in range(3):
        if aterts[ti] <= r["gbdt_auc"] < aterts[ti+1] or (ti == 2 and r["gbdt_auc"] == aterts[-1]):
            t = ti
            break
    r["auc_tertile"] = t

# --- Stratified selection: 4 drugs per IF quintile, balance tertiles ---
# Must include all 5 pilots (they are already-run observed data). Fill remaining
# 15 with confirmatory extension, 4 per quintile, spread across AUC tertiles.
selected = []
# Ensure pilots first (deterministic)
for r in sorted(frame, key=lambda x: x["drug"]):
    if r["is_pilot"]:
        selected.append(r)

# Now fill to 4 per quintile, preferring non-pilot, balancing AUC tertiles
rng = np.random.default_rng(SEED)
remaining = [r for r in frame if not r["is_pilot"]]
# deterministic order: shuffle with seed once, then greedy fill
rng.shuffle(remaining)
# count per quintile already
per_q = {qi: sum(1 for r in selected if r["quintile"] == qi) for qi in range(5)}
tert_cnt = {ti: sum(1 for r in selected if r["auc_tertile"] == ti) for ti in range(3)}

for r in remaining:
    q, t = r["quintile"], r["auc_tertile"]
    if per_q[q] >= N_PER_QUINTILE:
        continue
    selected.append(r)
    per_q[q] += 1
    tert_cnt[t] += 1

# Replacement rule (deterministic): if any quintile still under 4 after greedy,
# fill from the next-lowest-quintile pool, then lowest-AUC-underserved; log each.
replacement_log = []
for q in range(5):
    while per_q[q] < N_PER_QUINTILE:
        # find non-selected drug closest in IF (deterministic tie-break by name)
        candidates = [r for r in frame if r not in selected]
        candidates.sort(key=lambda r: (abs(r["if_full"] - edges[q+1] / (q+1 or 1)), r["drug"]))
        if not candidates:
            break
        pick = candidates[0]
        selected.append(pick)
        per_q[q] += 1
        tert_cnt[pick["auc_tertile"]] += 1
        replacement_log.append({"quintile": q, "replacement_for": None,
                                "picked": pick["drug"], "reason": "underfilled_quintile"})

selected.sort(key=lambda x: (x["quintile"], x["drug"]))

# --- Manifest ---
now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
manifest = {
    "format": "icbinb_revision_d3_frozen_selection.v1",
    "created": now,
    "generated_by": "scripts/freeze_d3_selection.py",
    "seed": SEED,
    "gate_threshold": GATE,
    "sampling_frame_n": len(frame),
    "total_selected": len(selected),
    "per_quintile": N_PER_QUINTILE,
    "pilot_drugs": PILOT_PARP,
    "confirmatory_extension_n": len([r for r in selected if not r["is_pilot"]]),
    "if_quintile_edges": [round(float(e), 6) for e in edges],
    "auc_tertile_edges": [round(float(e), 6) for e in aterts],
    "replacement_rule": "Greedy fill to 4 per IF quintile; if underfilled, take "
                        "closest-IF unselected drug, tie-broken by name. Logged below.",
    "replacement_log": replacement_log,
    "pre_quantum_inputs": "IF (full, shap 0.52.0) + GBDT AUC only; no quantum results viewed",
    "selected": [],
}
# serialize selected (strip runtime attrs for hash stability)
for r in selected:
    manifest["selected"].append({
        "drug": r["drug"], "class": r["class"], "is_pilot": r["is_pilot"],
        "if_full": round(r["if_full"], 6), "gbdt_auc": round(r["gbdt_auc"], 6),
        "quintile": r["quintile"], "auc_tertile": r["auc_tertile"], "n": r["n"],
    })

# content-address
canon = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
manifest["sha256"] = hashlib.sha256(canon.encode()).hexdigest()

out = REPO / "benchmarks" / "results" / "icbinb_d3_frozen_selection_20drug.json"
json.dump(manifest, open(out, "w"), indent=2)

print(f"WROTE {out}")
print(f"created: {now}  sha256: {manifest['sha256']}")
print(f"total={len(selected)}  pilots={sum(1 for r in selected if r['is_pilot'])} "
      f"extension={sum(1 for r in selected if not r['is_pilot'])}")
for q in range(5):
    qs = [r for r in selected if r["quintile"] == q]
    print(f"  quintile {q}: {len(qs)} -> {', '.join(r['drug'] for r in qs)}")
print("\nPer AUC tertile:", {ti: sum(1 for r in selected if r['auc_tertile']==ti) for ti in range(3)})
print("\nNOT selected (frame minus selected):")
sel_drugs = {r["drug"] for r in selected}
for r in sorted(frame, key=lambda x: x["drug"]):
    if r["drug"] not in sel_drugs:
        print(f"  {r['drug']}  IF={r['if_full']:.4f} auc={r['gbdt_auc']:.4f} q={r['quintile']}")

# Also write a CSV for quick review
with open(REPO / "benchmarks" / "results" / "icbinb_d3_frozen_selection_20drug.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["drug", "class", "is_pilot", "if_full", "gbdt_auc", "quintile", "auc_tertile", "n"])
    for r in selected:
        w.writerow([r["drug"], r["class"], r["is_pilot"], round(r["if_full"],5),
                    round(r["gbdt_auc"],5), r["quintile"], r["auc_tertile"], r["n"]])
print("WROTE", "benchmarks/results/icbinb_d3_frozen_selection_20drug.csv")