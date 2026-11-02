#!/usr/bin/env python
"""fetch_brca_receptor_status.py — ER/PR/HER2 from GDC BCR XML clinical supplements.

Why this exists
---------------
The GDC `cases` JSON API does NOT expose receptor status. All 770 fields of
`/cases/_mapping` contain zero ER/PR/HER2 entries -- a scraper pointed at that
endpoint silently returns empty strings for every patient (this is exactly how
`brca_clinical_labels.csv` ended up 0/139 filled while looking well-formed).

Receptor status lives only in the per-patient BCR XML Clinical Supplement, under:
    breast_carcinoma_estrogen_receptor_status
    breast_carcinoma_progesterone_receptor_status
    lab_proc_her2_neu_immunohistochemistry_receptor_status

Each XML holds the patient record plus follow-up records; follow-ups usually
carry empty receptor fields, so we take the FIRST non-empty value per field.

Usage
-----
    python scripts/fetch_brca_receptor_status.py \
        --patients ../data/paired_cohort/tcia_brca_patients.txt \
        --out ../data/paired_cohort/brca_receptor_status.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

FILES_EP = "https://api.gdc.cancer.gov/files"
DATA_EP = "https://api.gdc.cancer.gov/data"

FIELDS = {
    "breast_carcinoma_estrogen_receptor_status": "ER_Status",
    "breast_carcinoma_progesterone_receptor_status": "PR_Status",
    "lab_proc_her2_neu_immunohistochemistry_receptor_status": "HER2_Status",
}
# IHC is equivocal at 2+; FISH is the tiebreak and is worth carrying along
EXTRA = {
    "lab_procedure_her2_neu_in_situ_hybrid_outcome_type": "HER2_FISH",
    "er_level_cell_percentage_category": "ER_pct",
    "progesterone_receptor_level_cell_percent_category": "PR_pct",
}


def post(url: str, payload: dict, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code in (400, 404):
                raise
            wait = 2 ** a
            print(f"  retry {a+1}/{retries} after {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"POST {url} failed after {retries} attempts")


def get_bytes(url: str, retries: int = 4) -> bytes:
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** a
            print(f"  retry {a+1}/{retries} after {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"GET {url} failed")


def find_clinical_xml(patients: list[str]) -> dict[str, str]:
    """patient barcode -> clinical supplement file_id (batched, not one call each)."""
    out: dict[str, str] = {}
    B = 50
    for i in range(0, len(patients), B):
        chunk = patients[i:i + B]
        res = post(FILES_EP, {
            "filters": {"op": "and", "content": [
                {"op": "in", "content": {"field": "cases.submitter_id",
                                         "value": chunk}},
                {"op": "in", "content": {"field": "data_format",
                                         "value": ["BCR XML"]}},
                {"op": "in", "content": {"field": "data_type",
                                         "value": ["Clinical Supplement"]}},
            ]},
            "fields": "file_id,file_name,cases.submitter_id",
            "size": str(len(chunk) * 4),
            "format": "JSON",
        })
        for h in res["data"]["hits"]:
            name = h.get("file_name", "")
            if ".omf." in name:            # omics metadata file, not clinical
                continue
            for c in h.get("cases", []):
                sid = c.get("submitter_id")
                if sid and sid not in out:
                    out[sid] = h["file_id"]
        print(f"  mapped {len(out)}/{len(patients)}")
        time.sleep(0.3)
    return out


def parse_xml(blob: bytes) -> dict[str, str]:
    root = ET.fromstring(blob)
    want = {**FIELDS, **EXTRA}
    got: dict[str, str] = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in want:
            val = (el.text or "").strip()
            key = want[tag]
            if val and key not in got:      # first non-empty wins
                got[key] = val
    return got


def derive_subtype(r: dict[str, str]) -> str:
    """IHC surrogate subtype. NOT PAM50 -- but it is label-independent of RNA."""
    er = r.get("ER_Status", "").lower()
    pr = r.get("PR_Status", "").lower()
    h2 = r.get("HER2_Status", "").lower()
    fish = r.get("HER2_FISH", "").lower()

    if h2.startswith("equivocal"):
        if fish.startswith("positive"):
            h2 = "positive"
        elif fish.startswith("negative"):
            h2 = "negative"

    hr = er.startswith("positive") or pr.startswith("positive")
    hr_known = er.startswith(("positive", "negative")) or \
        pr.startswith(("positive", "negative"))
    h2_known = h2.startswith(("positive", "negative"))
    if not (hr_known and h2_known):
        return ""
    if hr and h2.startswith("positive"):
        return "HR+/HER2+"
    if hr:
        return "HR+/HER2-"
    if h2.startswith("positive"):
        return "HR-/HER2+"
    return "TripleNegative"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True,
                    help="one TCGA-XX-XXXX barcode per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/gdc_clinical_xml")
    a = ap.parse_args()

    patients = [l.strip() for l in open(a.patients) if l.strip()]
    print(f"[1/3] {len(patients)} patients")

    print("[2/3] locating clinical XML file_ids ...")
    fmap = find_clinical_xml(patients)
    missing = [p for p in patients if p not in fmap]
    if missing:
        print(f"  no clinical XML for {len(missing)}: {missing[:5]}")

    os.makedirs(a.cache, exist_ok=True)
    print("[3/3] downloading + parsing ...")
    rows = []
    for i, p in enumerate(patients, 1):
        fid = fmap.get(p)
        rec = {"PatientId": p, "ER_Status": "", "PR_Status": "",
               "HER2_Status": "", "HER2_FISH": "", "ER_pct": "", "PR_pct": "",
               "IHC_Subtype": ""}
        if fid:
            cp = os.path.join(a.cache, f"{p}.xml")
            if os.path.exists(cp) and os.path.getsize(cp) > 0:
                blob = open(cp, "rb").read()
            else:
                blob = get_bytes(f"{DATA_EP}/{fid}")
                with open(cp, "wb") as fh:
                    fh.write(blob)
                time.sleep(0.2)
            try:
                rec.update(parse_xml(blob))
            except ET.ParseError as e:
                print(f"  {p}: XML parse error {e}")
            rec["IHC_Subtype"] = derive_subtype(rec)
        rows.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(patients)}")

    cols = ["PatientId", "ER_Status", "PR_Status", "HER2_Status",
            "HER2_FISH", "ER_pct", "PR_pct", "IHC_Subtype"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    n = len(rows)
    print(f"\nwrote {a.out}")
    for c in ("ER_Status", "PR_Status", "HER2_Status", "IHC_Subtype"):
        filled = sum(1 for r in rows if r.get(c))
        print(f"  {c:12s} {filled:3d}/{n} ({filled/n*100:.1f}%)")

    import collections
    st = collections.Counter(r["IHC_Subtype"] for r in rows if r["IHC_Subtype"])
    print("\n  subtype distribution:")
    for k, v in st.most_common():
        print(f"    {k:16s} {v:3d}  ({v/max(1,sum(st.values()))*100:.1f}%)")
    if st:
        maj = max(st.values()) / sum(st.values())
        print(f"\n  majority baseline = {maj:.3f}")
    if sum(st.values()) == 0:
        print("\n  !! nothing extracted -- do not proceed on this file")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
