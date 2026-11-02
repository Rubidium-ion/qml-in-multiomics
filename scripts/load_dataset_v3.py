#!/usr/bin/env python3
"""load_dataset_v3.py — feature builder with a pathway whitelist that survives
variance filtering.

Motivation (authors, 2026-08-08)
------------------------------
v1/v2 selected features by top-variance alone. Measured on TCGA-BRCA n=138 that
filter ranks BRCA1 at 12653, PARP1 at 30618, PALB2 at 31523 -- every gene a
PARP/BRCA synthetic-lethality question depends on falls outside a top-5000 cut,
despite healthy expression (PARP1 mean log1p 9.68). Housekeeping DNA-repair
genes are stably expressed *by construction*, so variance is the wrong
instrument for selecting them.

v3 therefore builds the feature space as:

    features = whitelist_present  UNION  top_k_by_variance

The whitelist is forced in regardless of variance rank. `build_features` returns
a report naming which whitelist genes were found, which were missing from the
assay, and what each one's variance rank was -- so a silent drop is impossible.

Fold discipline
---------------
`build_features` is a *pure* function of the matrix it is handed. Call it inside
train() on training rows only; never on the full cohort. It does not fit any
scaler or projection -- downstream algorithms own that (see algorithms/_preprocess.py).

Status: loader skeleton. `load_gdsc_response` reads GDSC's published CSV/TSV
directly (no PyTDC). It refuses to guess a binarization rule -- `method` is a
required argument -- because that choice changes class balance and therefore
interaction_fraction, and must be fixed before anyone sees a result.
"""
#
# WARNING (authors, verified 2026-08-08)
# -----------------------------------
# `build_features` is only meaningful when handed a FULL gene matrix.
# data/brca_pr_v1/X.npy is 138x5000 -- it is ALREADY the top-variance subset
# produced by the v1 pipeline. Running the whitelist rescue on it reports
# whitelist_found=0 / n_rescued=0. That zero does NOT mean the genes are
# absent from the assay; it means they were deleted upstream before this file
# ever saw them. All 38 whitelist symbols ARE present in the raw STAR count
# matrices (60660 genes/sample) under data/rnaseq_star_counts_v2/.
#
# So: rebuild features from the 60660-gene matrix. Never from brca_pr_v1.
# build_features() now refuses pre-filtered matrices unless explicitly told.
#
from __future__ import annotations

import json
import os
import re
from typing import Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WHITELIST = os.path.join(REPO, "data", "hr_pathway_whitelist.txt")


# --------------------------------------------------------------------------
# whitelist
# --------------------------------------------------------------------------
def load_whitelist(path: str = DEFAULT_WHITELIST) -> list[str]:
    """Read gene symbols, ignoring '#' comments and blank lines."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"whitelist not found: {path}. Synthetic-lethality feature spaces "
            f"must not fall back to variance-only selection silently.")
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    # de-dupe, preserve order
    seen = set()
    return [g for g in out if not (g in seen or seen.add(g))]


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------
def build_features(X: np.ndarray,
                   gene_names: Sequence[str],
                   top_k: int = 5000,
                   whitelist: Sequence[str] | None = None,
                   allow_prefiltered: bool = False) -> dict:
    """Union of variance-ranked top_k and the pathway whitelist.

    Returns dict with:
        idx           column indices into X, ordered
        names         gene symbols for those columns
        report        provenance: found / missing / variance ranks

    Raises if X looks already variance-filtered (columns <= top_k), because
    rescuing a whitelist from a matrix it was already deleted from silently
    yields an empty rescue and a false clean bill of health.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != len(gene_names):
        raise ValueError(
            f"gene_names length {len(gene_names)} != X columns {X.shape[1]}")
    if not allow_prefiltered and X.shape[1] <= int(top_k):
        raise ValueError(
            f"X has {X.shape[1]} columns but top_k={top_k}: this matrix is "
            f"already variance-filtered, so the whitelist has nothing to "
            f"rescue and would silently report 0 found. Rebuild from the full "
            f"gene matrix (60660-gene STAR counts). Override with "
            f"allow_prefiltered=True only if that is genuinely intended.")
    if whitelist is None:
        whitelist = load_whitelist()

    var = X.var(axis=0)
    order = np.argsort(var)[::-1]
    rank = np.empty(len(var), dtype=int)
    rank[order] = np.arange(len(var))

    k = min(int(top_k), X.shape[1])
    top_idx = order[:k]

    sym_to_cols: dict[str, list[int]] = {}
    for i, g in enumerate(gene_names):
        sym_to_cols.setdefault(g, []).append(i)

    forced, found, missing = [], [], []
    for g in whitelist:
        cols = sym_to_cols.get(g)
        if not cols:
            missing.append(g)
            continue
        for c in cols:
            found.append({"gene": g, "col": int(c),
                          "variance_rank": int(rank[c]),
                          "in_top_k": bool(rank[c] < k)})
            forced.append(c)

    idx = np.array(sorted(set(top_idx.tolist()) | set(forced)), dtype=int)
    rescued = [f for f in found if not f["in_top_k"]]

    report = {
        "top_k": k,
        "n_features": int(len(idx)),
        "whitelist_size": len(whitelist),
        "whitelist_found": len(found),
        "whitelist_missing": missing,
        "rescued_by_whitelist": rescued,
        "n_rescued": len(rescued),
        "detail": found,
    }
    return {"idx": idx,
            "names": [gene_names[i] for i in idx],
            "report": report}


# --------------------------------------------------------------------------
# GDSC2 ingestion -- CSV mode (no PyTDC dependency)
# --------------------------------------------------------------------------
# authors asked to drop PyTDC and read GDSC's published tables directly. That is
# the better call regardless of environment: pinned files are reproducible, a
# live API is not, and one less dependency is one less thing to version-drift.

# Column-name candidates. GDSC has renamed these across releases, so accept the
# known aliases rather than hard-coding one release's schema.
_COL_ALIASES = {
    "drug":      ["DRUG_NAME", "drug_name", "Drug Name", "DRUG"],
    "cell":      ["COSMIC_ID", "cosmic_id", "CELL_LINE_NAME", "cell_line_name",
                  "SANGER_MODEL_ID", "Cell Line Name"],
    "ln_ic50":   ["LN_IC50", "ln_ic50"],
    "ic50":      ["IC50", "ic50", "IC50_PUBLISHED"],
    "z_score":   ["Z_SCORE", "z_score", "ZSCORE"],
}


def _resolve(header: Sequence[str], key: str) -> str | None:
    for cand in _COL_ALIASES[key]:
        if cand in header:
            return cand
    return None


def binarize_response(values: np.ndarray, method: str,
                      threshold: float | None = None) -> np.ndarray:
    """Continuous drug response -> binary responder label.

    method:
      'zscore'   responder if z < threshold   (default -1.5; lower = more sensitive)
      'quantile' responder if value <= q-th percentile (default 25)
      'absolute' responder if value <= threshold (same units as `values`)

    The choice materially changes class balance and therefore
    interaction_fraction, so it must be fixed *before* looking at results.
    There is deliberately no default method.
    """
    v = np.asarray(values, dtype=float)
    if method == "zscore":
        t = -1.5 if threshold is None else float(threshold)
        z = (v - np.nanmean(v)) / np.nanstd(v)
        return (z < t).astype(int)
    if method == "quantile":
        q = 25.0 if threshold is None else float(threshold)
        return (v <= np.nanpercentile(v, q)).astype(int)
    if method == "absolute":
        if threshold is None:
            raise ValueError("method='absolute' requires an explicit threshold")
        return (v <= float(threshold)).astype(int)
    raise ValueError(f"unknown binarization method: {method!r} "
                     f"(expected 'zscore' | 'quantile' | 'absolute')")


def load_gdsc_response(csv_path: str,
                       drug: str,
                       method: str,
                       threshold: float | None = None,
                       value_col: str | None = None) -> dict:
    """Load GDSC drug response from a published CSV/TSV and binarize it.

    Parameters
    ----------
    csv_path : GDSC2 fitted dose-response export (csv or tsv, optionally .gz)
    drug     : e.g. 'Olaparib', 'Talazoparib' (case-insensitive exact match)
    method   : passed to binarize_response; no default, must be chosen
    value_col: force a response column; otherwise LN_IC50 -> Z_SCORE -> IC50

    Returns dict with cell ids, continuous values, binary labels, and a
    provenance record of exactly which column and rule were used.
    """
    import csv as _csv
    import gzip

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"GDSC response table not found: {csv_path}\n"
            f"Download the GDSC2 fitted dose-response export and pass its path.")

    opener = gzip.open if csv_path.endswith(".gz") else open
    with opener(csv_path, "rt", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        rows = list(_csv.DictReader(fh, delimiter=delim))

    if not rows:
        raise ValueError(f"no rows parsed from {csv_path}")
    header = list(rows[0].keys())

    dcol = _resolve(header, "drug")
    ccol = _resolve(header, "cell")
    if dcol is None or ccol is None:
        raise ValueError(
            f"could not locate drug/cell columns in {header[:12]}... "
            f"pass a GDSC dose-response export, or extend _COL_ALIASES.")

    vcol = value_col or next(
        (c for c in (_resolve(header, "ln_ic50"),
                     _resolve(header, "z_score"),
                     _resolve(header, "ic50")) if c), None)
    if vcol is None:
        raise ValueError(f"no response column (LN_IC50/Z_SCORE/IC50) in {header[:12]}")

    want = drug.strip().lower()
    hit = [r for r in rows if (r.get(dcol) or "").strip().lower() == want]
    if not hit:
        seen = sorted({(r.get(dcol) or "").strip() for r in rows})
        near = [d for d in seen if want[:4] in d.lower()][:10]
        raise ValueError(
            f"drug {drug!r} not found in {csv_path}. "
            f"{len(seen)} distinct drugs present; similar: {near or 'none'}")

    cells, vals = [], []
    for r in hit:
        raw = (r.get(vcol) or "").strip()
        if raw in ("", "NA", "NaN", "null"):
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
        cells.append((r.get(ccol) or "").strip())

    if not vals:
        raise ValueError(f"drug {drug!r} matched {len(hit)} rows but none had a "
                         f"parseable {vcol}")

    v = np.asarray(vals, dtype=float)
    y = binarize_response(v, method=method, threshold=threshold)
    pos = int(y.sum())

    return {
        "cell_ids": cells,
        "values": v,
        "y": y,
        "provenance": {
            "csv_path": os.path.abspath(csv_path),
            "drug": drug,
            "n_cell_lines": int(len(v)),
            "response_column": vcol,
            "cell_column": ccol,
            "binarization": {"method": method, "threshold": threshold},
            "n_responder": pos,
            "n_non_responder": int(len(y) - pos),
            "responder_rate": round(pos / len(y), 4),
            "majority_baseline": round(max(pos, len(y) - pos) / len(y), 4),
        },
    }


if __name__ == "__main__":
    # Self-check against the existing TCGA cohort: demonstrates that whitelist
    # genes which variance filtering would discard are actually recovered.
    import glob

    # Must be the FULL 60660-gene matrix. brca_pr_v1 is already top-5000
    # filtered and will (correctly) trip the allow_prefiltered guard.
    full = os.path.join(REPO, "data", "tcga_brca_full")
    names_p = os.path.join(full, "gene_names.json")
    if not os.path.exists(names_p):
        raise SystemExit(
            "data/tcga_brca_full missing -- run scripts/build_expression_matrix.py "
            "first. Do NOT substitute data/brca_pr_v1: it is pre-filtered and "
            "the whitelist rescue there is a no-op.")

    X = np.load(os.path.join(full, "X.npy"))
    names = json.load(open(names_p))["gene_names"]
    wl = load_whitelist()
    res = build_features(X, names, top_k=5000, whitelist=wl)
    r = res["report"]
    print(f"whitelist genes            : {r['whitelist_size']}")
    print(f"present in this assay      : {r['whitelist_found']}")
    print(f"absent from assay          : {len(r['whitelist_missing'])}")
    print(f"rescued (would be dropped) : {r['n_rescued']}")
    print(f"final feature count        : {r['n_features']}")
    if r["whitelist_missing"]:
        print(f"  missing: {', '.join(r['whitelist_missing'])}")
    for d in r["rescued_by_whitelist"][:10]:
        print(f"  rescued {d['gene']:8s} variance_rank {d['variance_rank']}")
