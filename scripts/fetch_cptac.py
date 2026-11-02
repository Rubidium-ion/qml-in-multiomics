#!/usr/bin/env python3
"""Fetch CPTAC imaging (TCIA/NBIA) + omics (GDC) into the layout qmlmo expects.

CPTAC lives in TWO separate systems with two separate auth stories:

  imaging  -> TCIA / NBIA   services.cancerimagingarchive.net
  omics    -> GDC           api.gdc.cancer.gov

A GDC token does NOT authenticate against TCIA and vice versa. Most of what we
need is open access in both, so the default path uses no credentials at all.

CREDENTIAL HANDLING
  Never pass a token on the command line -- it lands in shell history, `ps`
  output and CI logs. This script reads a token only from:
      $GDC_TOKEN_FILE   path to the .txt downloaded from the GDC portal
  and only when --controlled is passed.

Usage:
    # 1. what cases have BOTH imaging and expression?
    python scripts/fetch_cptac.py match --collection CPTAC-LUAD --project CPTAC-3

    # 2. imaging: write a .tcia manifest, open it with NBIA Data Retriever
    python scripts/fetch_cptac.py imaging --collection CPTAC-LUAD \
        --modality CT --out ../data/cptac_luad

    # 3. omics: download open-access expression via the GDC API
    python scripts/fetch_cptac.py omics --project CPTAC-3 \
        --out ../data/cptac_luad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
GDC = "https://api.gdc.cancer.gov"
UA = {"User-Agent": "qmlmo-fetch/0.1"}


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         timeout: int = 120):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = e.read()[:400].decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} for {url}\n{body}") from None


def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: int = 300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={**UA, "Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = e.read()[:400].decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} for {url}\n{body}") from None


# --------------------------------------------------------------- TCIA imaging
def tcia_series(collection: str, modality: str | None) -> list[dict]:
    p = {"Collection": collection, "format": "json"}
    if modality:
        p["Modality"] = modality
    return json.loads(_get(f"{TCIA}/getSeries", p) or b"[]")


def cmd_imaging(args) -> int:
    series = tcia_series(args.collection, args.modality)
    if not series:
        raise SystemExit(
            f"no series for collection={args.collection!r} modality={args.modality!r}. "
            "Check the collection name at https://www.cancerimagingarchive.net/browse-collections/"
        )

    subjects = sorted({s.get("PatientID", "") for s in series} - {""})
    os.makedirs(args.out, exist_ok=True)

    # NBIA Data Retriever manifest. We do NOT stream DICOM over the plain API:
    # the retriever handles resume, integrity and rate limits properly.
    man = os.path.join(args.out, f"{args.collection}_{args.modality or 'ALL'}.tcia")
    with open(man, "w") as fh:
        fh.write("downloadServerUrl=https://services.cancerimagingarchive.net/nbia-api/services/v2/getImage\n")
        fh.write("includeAnnotation=true\nnoOfrRetry=4\ndatabasketId=qmlmo\n")
        fh.write("manifestVersion=3.0\nListOfSeriesToDownload=\n")
        for s in series:
            fh.write(s["SeriesInstanceUID"] + "\n")

    meta = os.path.join(args.out, "tcia_series.json")
    with open(meta, "w") as fh:
        json.dump(series, fh, indent=2)

    print(f"[tcia] collection={args.collection} modality={args.modality or 'ALL'}")
    print(f"[tcia] {len(series)} series across {len(subjects)} subjects")
    print(f"[tcia] manifest -> {man}")
    print(f"[tcia] metadata -> {meta}")
    print("\nNext: open the .tcia file with NBIA Data Retriever, and set its output to")
    print(f"      {os.path.join(os.path.abspath(args.out), 'ct')}")
    print("      with 'Descriptive Directory Name' so you get one dir per subject.")
    return 0


# ------------------------------------------------------------------ GDC omics
def gdc_files(project: str, data_type: str, workflow: str | None,
              token: str | None) -> list[dict]:
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "files.data_type", "value": [data_type]}},
    ]}
    if workflow:
        filters["content"].append(
            {"op": "in", "content": {"field": "files.analysis.workflow_type",
                                     "value": [workflow]}})

    payload = {
        "filters": filters,
        "fields": "file_id,file_name,data_type,cases.submitter_id,"
                  "cases.samples.sample_type,analysis.workflow_type",
        "format": "JSON", "size": "20000",
    }
    hdr = {"X-Auth-Token": token} if token else {}
    raw = _post_json(f"{GDC}/files", payload, hdr)
    return json.loads(raw)["data"]["hits"]


def _read_token(controlled: bool) -> str | None:
    """Token from $GDC_TOKEN_FILE only. Never from argv."""
    if not controlled:
        return None
    path = os.environ.get("GDC_TOKEN_FILE")
    if not path:
        raise SystemExit(
            "--controlled needs $GDC_TOKEN_FILE pointing at the token .txt from\n"
            "  https://portal.gdc.cancer.gov/  (Login -> Download Token)\n"
            "Tokens are never accepted as a command-line argument: argv is visible\n"
            "in shell history and `ps`."
        )
    if not os.path.exists(path):
        raise SystemExit(f"$GDC_TOKEN_FILE points at {path}, which does not exist")
    if os.name == "posix":
        mode = os.stat(path).st_mode & 0o077
        if mode:
            raise SystemExit(
                f"{path} is group/world readable. Run: chmod 600 {path}")
    return open(path).read().strip()


def cmd_omics(args) -> int:
    token = _read_token(args.controlled)
    hits = gdc_files(args.project, args.data_type, args.workflow, token)
    if not hits:
        raise SystemExit(
            f"no files for project={args.project} data_type={args.data_type!r}. "
            "Open-access CPTAC-3 expression is 'Gene Expression Quantification'."
        )

    os.makedirs(args.out, exist_ok=True)
    man = os.path.join(args.out, "gdc_manifest.txt")
    with open(man, "w") as fh:
        fh.write("id\tfilename\n")
        for h in hits:
            fh.write(f"{h['file_id']}\t{h['file_name']}\n")

    # case -> file map, so expression rows can be keyed by CPTAC case id later
    cmap = {}
    for h in hits:
        for c in h.get("cases", []):
            sid = c.get("submitter_id")
            if sid:
                cmap.setdefault(sid, []).append(h["file_id"])
    with open(os.path.join(args.out, "gdc_case_map.json"), "w") as fh:
        json.dump(cmap, fh, indent=2)

    print(f"[gdc] project={args.project} data_type={args.data_type!r}")
    print(f"[gdc] {len(hits)} files across {len(cmap)} cases")
    print(f"[gdc] manifest -> {man}")
    print(f"[gdc] case map -> {os.path.join(args.out, 'gdc_case_map.json')}")
    print("\nNext: gdc-client download -m gdc_manifest.txt -d <dir>")
    if args.controlled:
        print("      (add --token-file \"$GDC_TOKEN_FILE\" for controlled files)")
    return 0


# ------------------------------------------------------------------- matching
def cmd_match(args) -> int:
    """The number that decides whether cross-modal fusion is even possible."""
    series = tcia_series(args.collection, args.modality)
    img_cases = sorted({s.get("PatientID", "") for s in series} - {""})

    hits = gdc_files(args.project, args.data_type, args.workflow, None)
    omics_cases = sorted({c.get("submitter_id", "")
                          for h in hits for c in h.get("cases", [])} - {""})

    both = sorted(set(img_cases) & set(omics_cases))

    print(f"[match] TCIA {args.collection} ({args.modality or 'ALL'}): "
          f"{len(img_cases)} cases")
    print(f"[match] GDC  {args.project} ({args.data_type}): {len(omics_cases)} cases")
    print(f"[match] BOTH modalities: {len(both)} cases")

    if not both:
        print("\n  Zero overlap. Before assuming the cohorts are disjoint, check the")
        print("  id format on each side -- TCIA PatientID and GDC submitter_id do not")
        print("  always agree on prefixes/suffixes.")
        print(f"  TCIA sample ids : {img_cases[:3]}")
        print(f"  GDC  sample ids : {omics_cases[:3]}")
    elif len(both) < 40:
        print(f"\n  Only {len(both)} paired cases. With 5-fold CV that is ~{len(both)//5} "
              "test samples per fold, which will not support a fusion-vs-single-modality\n"
              "  comparison. Consider pooling several CPTAC collections.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "paired_cases.json")
        with open(p, "w") as fh:
            json.dump({"collection": args.collection, "project": args.project,
                       "imaging_only": sorted(set(img_cases) - set(both)),
                       "omics_only": sorted(set(omics_cases) - set(both)),
                       "paired": both}, fh, indent=2)
        print(f"\n[match] -> {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CPTAC fetch helper (TCIA + GDC)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("imaging", help="write a TCIA/NBIA .tcia manifest")
    p.add_argument("--collection", required=True, help="e.g. CPTAC-LUAD")
    p.add_argument("--modality", default="CT", help="CT, MR, PT, or omit for all")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_imaging)

    p = sub.add_parser("omics", help="write a GDC manifest for gdc-client")
    p.add_argument("--project", required=True, help="e.g. CPTAC-3")
    p.add_argument("--data-type", default="Gene Expression Quantification")
    p.add_argument("--workflow", default="STAR - Counts")
    p.add_argument("--out", required=True)
    p.add_argument("--controlled", action="store_true",
                   help="include controlled-access files; needs $GDC_TOKEN_FILE")
    p.set_defaults(fn=cmd_omics)

    p = sub.add_parser("match", help="count cases having BOTH imaging and omics")
    p.add_argument("--collection", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--modality", default="CT")
    p.add_argument("--data-type", default="Gene Expression Quantification")
    p.add_argument("--workflow", default="STAR - Counts")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_match)

    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
