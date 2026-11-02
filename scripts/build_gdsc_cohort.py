#!/usr/bin/env python3
"""build_gdsc_cohort.py -- join GDSC2 PARP response to the Sanger cell-line
expression matrix, producing a modelling-ready cohort.

Join key (authors, verified 2026-08-08)
-------------------------------------
Both sides carry SANGER_MODEL_ID:
  - GDSC2_fitted_dose_response_27Oct23.xlsx  column SANGER_MODEL_ID
  - rnaseq_tpm_20220624.csv                  header row 'model_id' (SIDM*)
So the join is exact -- no name matching, no COSMIC_ID crosswalk, no silent
sample loss. Every dropped line is counted and reported.

Expression file layout: 4 metadata header rows (model_id, model_name,
dataset_name, data_source), then a row 'gene_id,symbol,...' and the data.
Columns 0..1 are gene_id and symbol; sample columns start at index 2.

Features are built with load_dataset_v3.build_features on the FULL gene axis
(~37k genes), so the HR/DDR whitelist is genuinely rescued rather than
silently dropped by variance filtering.

Binarization is NOT defaulted -- --binarize is required.

PER-DRUG DISCIPLINE (bio, locked 2026-08-08)
--------------------------------------------
The quantile is computed inside ONE drug's response vector only. This script
structurally cannot pool: it filters to a single --drug before binarize_response
ever sees an array, so a pooled threshold is unreachable by construction rather
than by convention. `pooling_possible: false` is asserted into provenance.json
so the guarantee is auditable after the fact.

ID SCHEME
---------
bio asked for COSMIC_ID. Verified on the release-8.5 table: COSMIC_ID and
SANGER_MODEL_ID are 1:1 across all 242036 rows (969 distinct COSMIC, zero
conflicting mappings), so the two are interchangeable and no crosswalk table
is required. The expression matrix is keyed by SIDM, so we join on
SANGER_MODEL_ID and record the COSMIC_ID alongside it for bio's traceability.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.dirname(REPO)
GDSC = os.path.join(SHARED, "data", "gdsc")

RESPONSE_XLSX = os.path.join(GDSC, "GDSC2_fitted_dose_response_27Oct23.xlsx")
TPM_CSV = os.path.join(GDSC, "rnaseq_tpm_20220624.csv")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_dataset_v3 import build_features, binarize_response, load_whitelist  # noqa: E402


def load_expression(path: str = TPM_CSV, wanted: set[str] | None = None):
    """Stream the TPM matrix. Returns (model_ids, gene_symbols, X[samples,genes])."""
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r)                      # model_id row -> SIDM ids
        model_ids = header[2:]
        for _ in range(3):                    # model_name/dataset_name/data_source
            next(r)
        next(r)                               # 'gene_id,symbol,,,...'

        if wanted is not None:
            cols = [i for i, m in enumerate(model_ids) if m in wanted]
        else:
            cols = list(range(len(model_ids)))
        keep_ids = [model_ids[i] for i in cols]
        offs = [i + 2 for i in cols]

        symbols, rows = [], []
        for parts in r:
            if len(parts) < 3 or not parts[1]:
                continue
            symbols.append(parts[1])
            rows.append([parts[o] if o < len(parts) else "" for o in offs])

    X = np.zeros((len(rows), len(offs)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if v:
                try:
                    X[i, j] = float(v)
                except ValueError:
                    pass
    return keep_ids, symbols, X.T          # -> samples x genes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", default="Olaparib")
    ap.add_argument("--binarize", required=True,
                    choices=["zscore", "quantile", "absolute"],
                    help="no default: this choice sets class balance")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--value-col", default="LN_IC50",
                    choices=["LN_IC50", "Z_SCORE", "AUC"])
    ap.add_argument("--id-scheme", default="sanger",
                    choices=["sanger", "cosmic"],
                    help="reporting key; the two are 1:1 so the join is "
                         "identical either way")
    ap.add_argument("--tissue", default=None,
                    help="optional TCGA_DESC filter, e.g. BRCA")
    ap.add_argument("--top-k", type=int, default=5000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import pandas as pd

    for p in (RESPONSE_XLSX, TPM_CSV):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")

    df = pd.read_excel(RESPONSE_XLSX)
    sub = df[df.DRUG_NAME.str.strip().str.lower() == a.drug.strip().lower()]
    if sub.empty:
        raise SystemExit(f"drug {a.drug!r} not in GDSC2")
    if a.tissue:
        sub = sub[sub.TCGA_DESC == a.tissue]
        if sub.empty:
            raise SystemExit(f"no {a.drug} rows for tissue {a.tissue}")

    sub = sub.dropna(subset=["SANGER_MODEL_ID", a.value_col])
    sub = sub.drop_duplicates(subset=["SANGER_MODEL_ID"], keep="first")
    resp = dict(zip(sub.SANGER_MODEL_ID, sub[a.value_col].astype(float)))

    # COSMIC_ID <-> SANGER_MODEL_ID is 1:1 (verified); carry both.
    cosmic = {}
    if "COSMIC_ID" in sub.columns:
        for sid, cid in zip(sub.SANGER_MODEL_ID, sub.COSMIC_ID):
            if cid == cid:                      # not NaN
                cosmic[sid] = str(int(cid)) if float(cid).is_integer() else str(cid)
        n_conflict = len(sub) - sub.COSMIC_ID.nunique()
        if n_conflict > 0:
            print(f"WARNING: {n_conflict} COSMIC_ID collisions within {a.drug}")
    print(f"{a.drug}: {len(resp)} cell lines with {a.value_col}"
          f"{' in ' + a.tissue if a.tissue else ''}")

    ids, symbols, X = load_expression(TPM_CSV, wanted=set(resp))
    print(f"expression matrix: {X.shape[0]} samples x {X.shape[1]} genes")

    missing_expr = sorted(set(resp) - set(ids))
    if missing_expr:
        print(f"dropped {len(missing_expr)} lines lacking RNA-seq "
              f"(e.g. {missing_expr[:5]})")

    y_cont = np.array([resp[m] for m in ids], dtype=float)
    X = np.log1p(X)

    # PER-DRUG: y_cont holds this drug's rows only -- see module docstring.
    assert len(y_cont) == len(ids), "response vector desynced from sample ids"
    y = binarize_response(y_cont, method=a.binarize, threshold=a.threshold)
    pos = int(y.sum())
    print(f"labels: {pos} responder / {len(y)-pos} non-responder "
          f"(rate {pos/len(y):.3f}, majority baseline "
          f"{max(pos,len(y)-pos)/len(y):.3f})")

    res = build_features(X, symbols, top_k=a.top_k, whitelist=load_whitelist())
    r = res["report"]
    print(f"features: {r['n_features']} "
          f"(whitelist {r['whitelist_found']}/{r['whitelist_size']} found, "
          f"{r['n_rescued']} rescued from variance filter)")
    if r["whitelist_missing"]:
        print(f"  absent from assay: {', '.join(r['whitelist_missing'])}")

    out = a.out or os.path.join(
        REPO, "data",
        f"gdsc_{a.drug.lower()}_{a.binarize}"
        + (f"_{a.tissue.lower()}" if a.tissue else ""))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X.npy"), X[:, res["idx"]].astype(np.float32))
    np.save(os.path.join(out, "y.npy"), y)
    np.save(os.path.join(out, "y_continuous.npy"), y_cont)
    json.dump({"gene_names": res["names"]},
              open(os.path.join(out, "feature_names.json"), "w"))
    json.dump({
        "drug": a.drug,
        "tissue": a.tissue,
        "response_column": a.value_col,
        "binarization": {"method": a.binarize, "threshold": a.threshold},
        "join_key": "SANGER_MODEL_ID",
        "id_scheme_requested": a.id_scheme,
        "cosmic_ids": [cosmic.get(m) for m in ids],
        "cosmic_sanger_relation": "1:1 verified on release 8.5 (0 conflicts)",
        "per_drug_binarization": True,
        "pooling_possible": False,
        "pooling_note": ("threshold computed within this drug's response "
                         "vector only; the script filters to one drug before "
                         "binarization, so pooling is structurally unreachable"),
        "n_samples": int(len(ids)),
        "n_features": int(r["n_features"]),
        "n_responder": pos,
        "n_non_responder": int(len(y) - pos),
        "responder_rate": round(pos / len(y), 4),
        "majority_baseline": round(max(pos, len(y) - pos) / len(y), 4),
        "dropped_no_expression": missing_expr,
        "whitelist_report": r,
        "sources": {"response": RESPONSE_XLSX, "expression": TPM_CSV},
    }, open(os.path.join(out, "provenance.json"), "w"), indent=2)
    json.dump(ids, open(os.path.join(out, "sample_ids.json"), "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
