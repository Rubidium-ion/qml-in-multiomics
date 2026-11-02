#!/usr/bin/env python3
"""build_expression_matrix.py -- assemble the FULL gene x sample matrix from
raw GDC STAR count files, so that pathway-whitelist rescue is measurable.

Why (authors, 2026-08-08)
-----------------------
data/brca_pr_v1/X.npy is 138x5000: already top-variance filtered. Running
load_dataset_v3.build_features on it reports whitelist_found=0, which reads
like "these genes aren't in the assay" but actually means "they were deleted
before this stage". Verified against the raw files: all 38 whitelist symbols
are present in the 60660-gene STAR matrices.

This script rebuilds from those raw files so the rescue is real.

Output: data/tcga_brca_full/{X.npy, gene_names.json, sample_ids.json}
  X is log1p(TPM), samples x genes, orientation matching brca_pr_v1.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.dirname(REPO)
RAW_DIR = os.path.join(SHARED, "data", "rnaseq_star_counts_v2")
OUT_DIR = os.path.join(REPO, "data", "tcga_brca_full")

VALUE_COL = "tpm_unstranded"


def _read_one(path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Return (gene_ids, gene_names, values) for one STAR count TSV."""
    gids, gnames, vals = [], [], []
    with open(path) as fh:
        header = None
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                try:
                    vi = header.index(VALUE_COL)
                except ValueError:
                    raise SystemExit(
                        f"{VALUE_COL} not in {path} header: {header}")
                continue
            if not parts or parts[0].startswith("N_"):
                continue          # N_unmapped / N_multimapping summary rows
            if len(parts) <= vi:
                continue
            raw = parts[vi]
            if raw == "":
                continue
            gids.append(parts[0])
            gnames.append(parts[1])
            vals.append(float(raw))
    return gids, gnames, np.asarray(vals, dtype=float)


def main() -> None:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.tsv")))
    if not files:
        raise SystemExit(f"no STAR count files under {RAW_DIR}")

    ref_ids, ref_names, first = _read_one(files[0])
    n_genes = len(ref_ids)
    print(f"reference sample : {os.path.basename(files[0])[:24]}  "
          f"{n_genes} genes")

    X = np.empty((len(files), n_genes), dtype=np.float32)
    X[0] = first
    sample_ids = [os.path.basename(files[0]).split("_")[0]]

    for i, f in enumerate(files[1:], start=1):
        gids, _, v = _read_one(f)
        if gids != ref_ids:
            raise SystemExit(
                f"gene axis mismatch in {os.path.basename(f)}: refusing to "
                f"stack matrices whose rows are not aligned.")
        X[i] = v
        sample_ids.append(os.path.basename(f).split("_")[0])

    # collapse duplicate barcodes (same patient sequenced twice)
    seen: dict[str, int] = {}
    keep = []
    for i, s in enumerate(sample_ids):
        if s not in seen:
            seen[s] = i
            keep.append(i)
    if len(keep) != len(sample_ids):
        print(f"dropped {len(sample_ids)-len(keep)} duplicate barcode(s)")
        X = X[keep]
        sample_ids = [sample_ids[i] for i in keep]

    X = np.log1p(X)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    json.dump({"gene_ids": ref_ids, "gene_names": ref_names},
              open(os.path.join(OUT_DIR, "gene_names.json"), "w"))
    json.dump(sample_ids, open(os.path.join(OUT_DIR, "sample_ids.json"), "w"))

    print(f"wrote {OUT_DIR}")
    print(f"  X            : {X.shape}  log1p({VALUE_COL})")
    print(f"  samples      : {len(sample_ids)}")

    # immediately prove the rescue works on this matrix
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_dataset_v3 import build_features, load_whitelist

    wl = load_whitelist()
    res = build_features(X, ref_names, top_k=5000, whitelist=wl)
    r = res["report"]
    print("\n-- whitelist rescue on FULL matrix --")
    print(f"  whitelist genes : {r['whitelist_size']}")
    print(f"  found in assay  : {r['whitelist_found']}")
    print(f"  missing         : {len(r['whitelist_missing'])} "
          f"{r['whitelist_missing'] or ''}")
    print(f"  rescued         : {r['n_rescued']}")
    print(f"  final features  : {r['n_features']}")
    for d in sorted(r["rescued_by_whitelist"],
                    key=lambda d: d["variance_rank"])[:12]:
        print(f"    {d['gene']:9s} variance_rank {d['variance_rank']}")


if __name__ == "__main__":
    main()
