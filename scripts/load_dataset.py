#!/usr/bin/env python3
"""
load_dataset.py — reproducible dataset loader for the QML-in-multiomics benchmark.

Normalizes every dataset to the v0 interface contract (see docs/algorithm_interface_spec.md):
    X : (n_samples, n_features) float32
    y : (n_samples,)            int32   (binary 0/1 or multi-class)

Currently implemented source: TCGA RNA-seq STAR gene counts (tumor vs normal).

Pipeline (fully scripted / reproducible):
    raw TSV per sample  ->  gene x sample count matrix
    ->  log1p(counts)
    ->  top-k variance gene filter
    ->  StandardScaler
    ->  PCA(n_components)
    ->  train/test split (stratified)

Label source (in priority order):
    1. --labels CSV  (columns: sample_id,label)  -- explicit sample sheet
    2. TCGA barcode heuristic from filename/aliquot: sample type code
       01-09 = tumor(1), 10-19 = normal(0). Requires a manifest mapping
       file-uuid -> barcode (--manifest). 
    3. --labels-npy  (pre-computed y.npy aligned to sorted TSV order)

Usage:
    python scripts/load_dataset.py \
        --source tcga_rnaseq \
        --input-dir ../data/tcga_brca/2026-07-17 \
        --labels-npy ../data/tcga_brca/2026-07-17/y.npy \
        --pca 32 --top-var-genes 5000 --test-ratio 0.25 --seed 42 \
        --out-dir data/tcga_brca_v1

Outputs (in --out-dir):
    X.npy y.npy X_train.npy X_test.npy y_train.npy y_test.npy
    feature_meta.json   (pipeline params + explained variance)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np


def _load_tcga_counts(input_dir: str, count_col: str = "unstranded"):
    """Read all *.rna_seq.augmented_star_gene_counts.tsv into (genes x samples).

    Returns (counts[n_genes, n_samples] float32, gene_ids[list], sample_files[list]).
    Samples are ordered by sorted filename for determinism.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*augmented_star_gene_counts.tsv")))
    if not files:
        raise FileNotFoundError(f"no STAR gene-count TSVs found in {input_dir}")

    gene_ids = None
    cols = []
    for fp in files:
        genes = []
        vals = []
        with open(fp) as fh:
            header = None
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if header is None:
                    header = parts
                    try:
                        ci = header.index(count_col)
                    except ValueError:
                        raise ValueError(f"count column '{count_col}' not in {fp} header {header}")
                    continue
                gid = parts[0]
                # skip STAR summary rows (N_unmapped etc.)
                if gid.startswith("N_"):
                    continue
                genes.append(gid)
                v = parts[ci]
                vals.append(float(v) if v not in ("", None) else 0.0)
        if gene_ids is None:
            gene_ids = genes
        elif genes != gene_ids:
            raise ValueError(f"gene ordering mismatch in {fp}; datasets not aligned")
        cols.append(np.asarray(vals, dtype=np.float32))

    counts = np.vstack(cols).T  # genes x samples
    return counts.astype(np.float32), gene_ids, files


def _labels_from_npy(path: str, n: int) -> np.ndarray:
    y = np.load(path).astype(np.int32).ravel()
    if y.shape[0] != n:
        raise ValueError(f"label count {y.shape[0]} != n_samples {n}")
    return y


def _labels_from_csv(path: str, sample_files: list[str], label_col: str = "label",
                     id_col: str | None = None):
    """CSV -> (y, classes).

    Accepts either integer labels or STRING class names (PAM50 subtypes:
    LumA/LumB/Basal/Her2/Normal, or ER/PR/HER2 status). Strings are encoded to
    contiguous ints in sorted order and the mapping is returned so it can be
    persisted in feature_meta.json -- without it a downstream "class 2" is
    unreadable.

    sample_id is matched against, in order: the file's uuid prefix, the full
    basename, and the TCGA patient barcode (TCGA-XX-XXXX) if present in the
    filename. The barcode path is what makes authors' GDC mapping table work
    directly, without renaming files.
    """
    import csv
    import re

    rows = []
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        if rdr.fieldnames is None:
            raise SystemExit(f"{path}: empty CSV")
        cols = {c.lower(): c for c in rdr.fieldnames}
        lc = cols.get(label_col.lower())
        if lc is None:
            raise SystemExit(
                f"{path}: no column {label_col!r}; available: {rdr.fieldnames}")
        if id_col:
            ic = cols.get(id_col.lower())
            if ic is None:
                raise SystemExit(
                    f"{path}: no column {id_col!r}; available: {rdr.fieldnames}")
        else:
            ic = next((cols[c] for c in
                       ("sample_id", "file_id", "patient", "patientid",
                        "barcode", "case_id", "submitter_id")
                       if c in cols), None)
            if ic is None:
                ic = rdr.fieldnames[0]
        for row in rdr:
            key = (row.get(ic) or "").strip()
            val = (row.get(lc) or "").strip()
            if key and val:
                rows.append((key, val))

    mapping = {k: v for k, v in rows}
    bc_re = re.compile(r"TCGA-[0-9A-Z]{2}-[0-9A-Z]{4}", re.I)
    by_barcode = {}
    for k, v in rows:
        m = bc_re.search(k)
        if m:
            by_barcode[m.group(0).upper()] = v

    raw = []
    missing = []
    for fp in sample_files:
        base = os.path.basename(fp)
        uuid = base.split(".")[0]
        hit = mapping.get(uuid) or mapping.get(base)
        if hit is None:
            m = bc_re.search(base)
            if m:
                hit = by_barcode.get(m.group(0).upper())
        if hit is None:
            missing.append(base)
        raw.append(hit)

    if missing:
        raise SystemExit(
            f"{len(missing)} sample(s) have no label in {path}, e.g. "
            f"{missing[:3]}. Label CSV must cover every file in --input-dir "
            f"(ids tried: uuid prefix, full basename, TCGA barcode)."
        )

    # int labels pass through; strings get encoded
    try:
        y = np.asarray([int(v) for v in raw], dtype=np.int32)
        classes = [str(c) for c in sorted(set(y.tolist()))]
    except ValueError:
        classes = sorted(set(raw))
        idx = {c: i for i, c in enumerate(classes)}
        y = np.asarray([idx[v] for v in raw], dtype=np.int32)
    return y, classes


def _check_labels(y, classes, min_per_class: int = 5):
    """Loud failure beats a silent single-class run."""
    counts = np.bincount(y, minlength=len(classes))
    dist = {classes[i]: int(counts[i]) for i in range(len(classes))}
    print(f"[labels] {len(y)} samples, {len(classes)} classes: {dist}")
    if len(classes) < 2:
        raise SystemExit(
            f"only one class present ({classes}) -- classification is undefined. "
            "If you filtered to Primary Tumor only, you need a subtype/status "
            "label, not tumor-vs-normal."
        )
    majority = float(counts.max() / counts.sum())
    print(f"[labels] majority baseline = {majority:.3f}")
    thin = {c: n for c, n in dist.items() if n < min_per_class}
    if thin:
        print(f"[labels] WARNING thin classes (<{min_per_class}): {thin} -- "
              "stratified CV folds may end up with zero test samples for these")
    return dist, majority


def build(args) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    if args.source != "tcga_rnaseq":
        raise NotImplementedError(f"source '{args.source}' not implemented yet")

    counts, gene_ids, sample_files = _load_tcga_counts(args.input_dir, args.count_col)
    n_samples = counts.shape[1]

    # labels
    label_classes = None
    if args.labels:
        y, label_classes = _labels_from_csv(
            args.labels, sample_files, args.label_col, args.id_col)
    elif args.labels_npy:
        y = _labels_from_npy(args.labels_npy, n_samples)
        label_classes = [str(c) for c in sorted(set(y.tolist()))]
    else:
        raise SystemExit("no label source: pass --labels CSV or --labels-npy")

    label_dist, majority_baseline = _check_labels(y, label_classes)

    # samples x genes
    X_raw = counts.T  # (n_samples, n_genes)

    # log1p stabilize
    X_log = np.log1p(X_raw)

    # top-variance gene filter
    variances = X_log.var(axis=0)
    k = min(args.top_var_genes, X_log.shape[1])
    top_idx = np.argsort(variances)[::-1][:k]
    X_top = X_log[:, top_idx]
    top_gene_ids = [gene_ids[i] for i in top_idx]

    # --pca 0  ->  emit RAW (log1p + top-var) features, no global scaler, no PCA.
    # This is the only mode that supports honest k-fold CV: every transform must
    # then happen inside each algorithm's train() / inside a fold Pipeline.
    if args.pca <= 0:
        X = X_top.astype(np.float32)
        n_comp = 0
        pca = None
    else:
        # scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_top)

        # PCA (cap components to n_samples-1)
        n_comp = min(args.pca, X_scaled.shape[0] - 1, X_scaled.shape[1])
        pca = PCA(n_components=n_comp, random_state=args.seed)
        X = pca.fit_transform(X_scaled).astype(np.float32)
    y = y.astype(np.int32)

    # split (stratified when possible)
    strat = y if len(np.unique(y)) > 1 and np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_ratio, random_state=args.seed, stratify=strat
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_tr.astype(np.float32))
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_te.astype(np.float32))
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_tr.astype(np.int32))
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_te.astype(np.int32))

    # --- angle-encoding companion features (for qsvm_angle) --------------------
    # Angle Encoding (RZ-based feature map) is 2*pi periodic: z-scored PCA values
    # with heavy tails (|x| > ~pi) cause phase wrapping in RZ((pi-x_i)(pi-x_j)),
    # destroying quantum-kernel separability. We therefore emit an optional set
    # of angle-scaled features, MinMax-mapped into [0, pi] (single RZ period).
    # Scaler is FIT ON TRAIN ONLY, then applied to test/full to avoid leakage.
    angle_meta = None
    if args.angle_scale:
        from sklearn.preprocessing import MinMaxScaler

        hi = float(args.angle_max)  # default pi
        ascaler = MinMaxScaler(feature_range=(0.0, hi))
        Xtr_a = ascaler.fit_transform(X_tr).astype(np.float32)
        Xte_a = ascaler.transform(X_te).astype(np.float32)
        X_a = ascaler.transform(X).astype(np.float32)
        # test-set values can fall slightly outside [0, hi]; clip back into the
        # single RZ period so no sample silently wraps.
        Xte_a = np.clip(Xte_a, 0.0, hi)
        X_a = np.clip(X_a, 0.0, hi)
        np.save(os.path.join(args.out_dir, "X_angle.npy"), X_a)
        np.save(os.path.join(args.out_dir, "X_train_angle.npy"), Xtr_a)
        np.save(os.path.join(args.out_dir, "X_test_angle.npy"), Xte_a)
        angle_meta = {
            "enabled": True,
            "range": [0.0, hi],
            "range_label": "[0, pi]" if abs(hi - np.pi) < 1e-6 else f"[0, {hi:.4f}]",
            "scaler": "MinMaxScaler(fit_on_train)",
            "n_features": int(X_a.shape[1]),
            "test_clipped": True,
            "note": ("angle-encoded features for qsvm_angle; svm_rbf/rf_baseline "
                     "should keep using the standard X_train/X_test PCA features"),
        }

    meta = {
        "source": args.source,
        "input_dir": os.path.abspath(args.input_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(n_samples),
        "label_classes": label_classes,
        "label_distribution": label_dist,
        "majority_baseline": majority_baseline,
        "n_classes": int(len(label_classes)),
        "n_genes_raw": int(counts.shape[0]),
        "count_col": args.count_col,
        "top_var_genes": int(k),
        "pca_components": int(n_comp),
        "explained_var_total": (float(pca.explained_variance_ratio_.sum())
                                if pca is not None else None),
        "explained_var_ratio": ([float(v) for v in pca.explained_variance_ratio_]
                                if pca is not None else []),
        "preprocessing": ("log1p + top-var filter only (raw; reduce inside fold)"
                          if pca is None else
                          "log1p + top-var filter + StandardScaler + PCA (pre-projected)"),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "class_balance": {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))},
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "stratified": strat is not None,
        "top_gene_ids_head": top_gene_ids[:20],
        "sample_files": [os.path.basename(f) for f in sample_files],
    }
    if angle_meta is not None:
        meta["angle_scale"] = angle_meta
    with open(os.path.join(args.out_dir, "feature_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reproducible dataset loader (v0 contract)")
    p.add_argument("--source", default="tcga_rnaseq")
    p.add_argument("--input-dir", default="../data/tcga_brca/2026-07-17",
                   help="STAR gene-count TSV dir (default: shared symlinked TCGA-BRCA)")
    p.add_argument("--count-col", default="unstranded",
                   help="STAR count column: unstranded / tpm_unstranded / fpkm_unstranded")
    p.add_argument("--labels", help="CSV with an id column and a label column")
    p.add_argument("--label-col", default="label",
                   help="label column name, e.g. pam50 / er_status")
    p.add_argument("--id-col", default=None,
                   help="id column name; auto-detected if omitted")
    p.add_argument("--labels-npy", help="pre-computed y.npy aligned to sorted TSV order")
    p.add_argument("--top-var-genes", type=int, default=5000)
    p.add_argument("--pca", type=int, default=32,
                   help="0 = skip StandardScaler+PCA and emit raw log1p top-var "
                        "genes (required for honest k-fold CV)")
    p.add_argument("--test-ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--angle-scale", action="store_true",
                   help="also emit X_angle/X_train_angle/X_test_angle MinMax-scaled "
                        "to [0, pi] for qsvm_angle (avoids RZ phase wrapping)")
    p.add_argument("--angle-max", type=float, default=float(np.pi),
                   help="upper bound of angle scaling range (default: pi)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    meta = build(args)
    print(json.dumps(meta, indent=2))
    print(f"\n[load_dataset] wrote 6 .npy + feature_meta.json -> {args.out_dir}", file=sys.stderr)
    if meta["pca_components"]:
        print(f"[load_dataset] explained_var_total={meta['explained_var_total']:.4f} "
              f"({meta['pca_components']} PCs, {meta['n_samples']} samples)",
              file=sys.stderr)
    else:
        print(f"[load_dataset] RAW mode: {meta['top_var_genes']} genes, "
              f"{meta['n_samples']} samples (no PCA -> k-fold CV is honest)",
              file=sys.stderr)
