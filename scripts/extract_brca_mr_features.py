#!/usr/bin/env python3
"""Batch offline feature extraction for TCGA-BRCA MR volumes.

Extracts first-order intensity radiomics for all 139 subjects and saves
a persistent feature matrix (CSV) to data/tcga_brca/brca_mr_features.csv.
Progress is saved incrementally so it can resume if interrupted.
"""
import os
import sys
import gc
import json
import csv
import numpy as np
import pandas as pd

# repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qmlmo.modalities.imaging_dicom import (
    _require_pydicom,
    _load_metadata,
    _pick_series_dir,
    _read_series,
    _first_order_features,
    _feature_names,
)

ROOT = "/mnt/d/nextcloud/data/medical-imaging/TCGA-BRCA/radiology/TCGA-BRCA/TCGA-BRCA"
META_CSV = "/mnt/d/nextcloud/data/medical-imaging/TCGA-BRCA/radiology/TCGA-BRCA/metadata.csv"
OUT_CSV = "data/tcga_brca/brca_mr_features.csv"
PROGRESS_JSON = "data/tcga_brca/brca_mr_extraction_progress.json"

PARAMS = {
    "want_modality": "MR",
    "zscore_volume": True,
    "min_slices": 8,
    "mask_percentile": 0,
}

def main():
    pydicom = _require_pydicom()
    meta_rows = _load_metadata(META_CSV, ROOT)
    
    # Target subjects from labels
    labels_df = pd.read_csv("data/tcga_brca/labels_brca_mr_paired.csv")
    target_subjects = list(labels_df["case_id"].dropna().unique())
    print(f"[MR-Batch] Total target subjects in cohort: {len(target_subjects)}")

    # Check progress
    done_features = {}
    if os.path.exists(OUT_CSV):
        try:
            df_existing = pd.read_csv(OUT_CSV)
            if "case_id" in df_existing.columns:
                for _, row in df_existing.iterrows():
                    cid = row["case_id"]
                    feats = [row[c] for c in _feature_names()]
                    done_features[cid] = feats
                print(f"[MR-Batch] Found existing cache with {len(done_features)} processed cases.")
        except Exception as e:
            print(f"[MR-Batch] Could not load existing CSV: {e}")

    feat_names = _feature_names()
    processed_count = 0

    for i, sid in enumerate(target_subjects, 1):
        if sid in done_features:
            continue

        subj_dir = os.path.join(ROOT, sid)
        if not os.path.isdir(subj_dir):
            print(f"[{i}/{len(target_subjects)}] {sid}: directory not found under ROOT, skipping.")
            continue

        series_dir = _pick_series_dir(ROOT, sid, "MR", meta_rows, PARAMS)
        if not series_dir:
            print(f"[{i}/{len(target_subjects)}] {sid}: no usable MR series found, skipping.")
            continue

        print(f"[{i}/{len(target_subjects)}] Extracting {sid} from {os.path.basename(series_dir)} ...", flush=True)
        try:
            vol, mod, vmeta = _read_series(pydicom, series_dir, PARAMS)
            if mod != "MR":
                print(f"  Warning: mod is {mod}, not MR. Skipping.")
                continue
            feats = _first_order_features(vol, PARAMS)
            done_features[sid] = feats
            processed_count += 1
            del vol
            gc.collect()

            # Save incrementally every 5 cases or on completion
            if True: # save each case
                rows = [{"case_id": k, **dict(zip(feat_names, v))} for k, v in done_features.items()]
                df_out = pd.DataFrame(rows)
                df_out.to_csv(OUT_CSV, index=False)
                print(f"  --> Saved checkpoint: {len(df_out)} cases in {OUT_CSV}", flush=True)
        except Exception as exc:
            print(f"  Error extracting {sid}: {exc}")

    # Final save
    rows = [{"case_id": k, **dict(zip(feat_names, v))} for k, v in done_features.items()]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[MR-Batch] Finished! Total extracted: {len(df_out)} / {len(target_subjects)} -> {OUT_CSV}")

if __name__ == "__main__":
    main()
