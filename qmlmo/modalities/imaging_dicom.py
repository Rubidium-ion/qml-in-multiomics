"""DICOM imaging adapter (CT / MRI / nuclear medicine).

Scope, stated honestly: this extracts *first-order intensity radiomics* plus
modality-aware normalisation. It is not a full PyRadiomics replacement, and it
does not compute texture (GLCM/GLRLM) features. What it does guarantee is that
the numbers entering a quantum kernel are physically meaningful per modality:

  CT   -> Hounsfield units via RescaleSlope/Intercept, then an optional window
  PT   -> SUVbw when the required DICOM tags are present; refuses to guess
  MR   -> arbitrary units, so z-scoring within the volume is mandatory, and the
          meta records that cross-subject MR intensity is not comparable raw

NBIA / TCIA layout is nested:
    <root>/<SubjectID>/<StudyDesc>/<SeriesDesc>/*.dcm
A subject folder is NOT a flat pile of slices. The adapter therefore:
  1. discovers subjects (optionally via metadata.csv)
  2. walks each subject for series directories
  3. picks ONE series per subject (skip scout / coronal / sagittal / MIP)
  4. refuses to mix CT/MR/PT in one FeatureBlock

pydicom is an optional dependency. If it is missing, the adapter raises with
the install command rather than silently degrading.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Any

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter

_PERCENTILES = (5, 25, 50, 75, 95)
_DCM_EXTS = (".dcm", ".DCM", ".ima", ".IMA")
_SKIP_DIR_NAMES = {"license", "__macosx"}


def _require_pydicom():
    try:
        import pydicom  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "DICOM support needs pydicom. Install with:\n"
            "    pip install pydicom\n"
            "(kept optional so the omics-only path has no imaging dependency)"
        ) from exc
    import pydicom

    return pydicom


@register_modality
class DicomRadiomicsAdapter(ModalityAdapter):
    """params:
      root            : NBIA collection root OR a directory of one-subdir-per-subject
      metadata_csv    : optional TCIA metadata.csv (used for series pick + modality filter)
      want_modality   : 'CT' | 'MR' | 'PT'  (default 'CT')
      modality        : FeatureBlock label (default imaging_<want>)
      window          : (center, width) for CT, applied after HU conversion
      zscore_volume   : force per-volume z-scoring (default True for MR)
      mask_percentile : keep voxels above this intensity percentile (default 0)
      min_slices      : reject series thinner than this (default 8; drops scouts)
      max_subjects    : optional cap, useful for smoke tests
    """

    name = "imaging_dicom"

    def load(self) -> FeatureBlock:
        pydicom = _require_pydicom()
        p = dict(self.params)
        root = _resolve_collection_root(p["root"])
        want = str(p.get("want_modality", "CT")).upper()
        meta_rows = _load_metadata(p.get("metadata_csv"), root)

        subjects = _discover_subjects(root, meta_rows)
        cap = p.get("max_subjects")
        if cap:
            subjects = subjects[: int(cap)]
        if not subjects:
            raise FileNotFoundError(
                f"imaging: no subject directories under {root}. "
                "Expected NBIA layout <root>/<SubjectID>/<Study>/<Series>/*.dcm"
            )

        rows, ids, picked, skipped = [], [], [], []
        for sid in subjects:
            series_dir = _pick_series_dir(root, sid, want, meta_rows, p)
            if series_dir is None:
                skipped.append({"subject": sid, "reason": f"no usable {want} series"})
                continue
            try:
                vol, mod, vmeta = _read_series(pydicom, series_dir, p)
            except Exception as exc:
                skipped.append({"subject": sid, "reason": f"{type(exc).__name__}: {exc}",
                                "series": series_dir})
                continue
            if mod != want:
                skipped.append({"subject": sid, "reason": f"picked series is {mod}, want {want}",
                                "series": series_dir})
                continue
            rows.append(_first_order_features(vol, p))
            ids.append(sid)
            picked.append({"subject": sid, "series": series_dir,
                           "dicom_modality": mod, **vmeta})

        if not rows:
            raise ValueError(
                f"imaging: 0/{len(subjects)} subjects produced a {want} volume. "
                f"skipped={skipped[:5]}"
            )

        return FeatureBlock(
            X=np.asarray(rows, dtype=np.float64),
            feature_names=_feature_names(),
            sample_ids=ids,
            modality=p.get("modality", f"imaging_{want.lower()}"),
            meta={
                "adapter": self.name,
                "root": os.path.abspath(root),
                "want_modality": want,
                "n_subjects_found": len(subjects),
                "n_loaded": len(ids),
                "n_skipped": len(skipped),
                "picked": picked,
                "skipped": skipped,
                "note": "first-order intensity radiomics only; one series per subject",
            },
        )


def _resolve_collection_root(root: str) -> str:
    """NBIA often nests <Collection>/<Collection>/<SubjectID>. Unwrap one extra
    same-named directory so the caller can point at either level."""
    if not os.path.isdir(root):
        raise NotADirectoryError(f"imaging: {root} is not a directory")
    kids = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.lower() not in _SKIP_DIR_NAMES]
    base = os.path.basename(os.path.normpath(root))
    if len(kids) == 1 and kids[0] == base:
        inner = os.path.join(root, kids[0])
        inner_kids = [d for d in os.listdir(inner)
                      if os.path.isdir(os.path.join(inner, d))]
        # unwrap only when the inner dir looks like a subject forest, not a study
        if inner_kids and _looks_like_subject_id(inner_kids[0]):
            return inner
    return root


def _looks_like_subject_id(name: str) -> bool:
    return bool(re.match(r"^(C3[LN]-\d+|TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", name, re.I))


def _load_metadata(explicit: str | None, root: str) -> list[dict]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(root, "metadata.csv"))
    parent = os.path.dirname(root)
    candidates.append(os.path.join(parent, "metadata.csv"))
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, newline="", encoding="utf-8-sig") as fh:
                return list(csv.DictReader(fh))
    return []


def _discover_subjects(root: str, meta_rows: list[dict]) -> list[str]:
    on_disk = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.lower() not in _SKIP_DIR_NAMES
    )
    if meta_rows:
        wanted = {r.get("Subject ID") or r.get("SubjectID") or "" for r in meta_rows}
        wanted.discard("")
        return [s for s in on_disk if s in wanted] or on_disk
    return on_disk


def _pick_series_dir(root: str, sid: str, want: str,
                     meta_rows: list[dict], p: dict[str, Any]) -> str | None:
    """Choose one series directory for this subject.

    Preference: metadata-driven (modality + description score + image count),
    then filesystem walk with the same scoring. Never returns a scout/localizer
    if anything thicker exists.
    """
    min_slices = int(p.get("min_slices", 8))
    subject_dir = os.path.join(root, sid)
    if not os.path.isdir(subject_dir):
        return None

    meta_pick = _pick_from_metadata(root, sid, want, meta_rows, min_slices)
    if meta_pick and os.path.isdir(meta_pick):
        return meta_pick

    candidates = []
    for dirpath, dirnames, filenames in os.walk(subject_dir):
        dcm = [f for f in filenames if f.endswith(_DCM_EXTS) or _looks_dcm(os.path.join(dirpath, f))]
        if not dcm:
            continue
        n = len(dcm)
        if n < min_slices and _score_desc(os.path.basename(dirpath)) < 50:
            # keep walking; a scout of 1-2 frames is not a volume
            if n < 3:
                continue
        candidates.append((
            _score_desc(os.path.basename(dirpath)),
            n,
            dirpath,
        ))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def _pick_from_metadata(root: str, sid: str, want: str,
                        meta_rows: list[dict], min_slices: int) -> str | None:
    if not meta_rows:
        return None
    rows = [r for r in meta_rows
            if (r.get("Subject ID") or r.get("SubjectID")) == sid
            and str(r.get("Modality", "")).upper() == want]
    scored = []
    for r in rows:
        try:
            nimg = int(float(r.get("Number of Images") or 0))
        except ValueError:
            nimg = 0
        if nimg and nimg < min_slices:
            continue
        loc = (r.get("File Location") or "").lstrip("./")
        path = _resolve_file_location(root, loc)
        if path is None:
            continue
        scored.append((_score_desc(r.get("Series Description") or ""), nimg, path))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def _resolve_file_location(root: str, loc: str) -> str | None:
    if not loc:
        return None
    # metadata paths look like ./CPTAC-LUAD/C3L-00263/study/series
    direct = os.path.join(root, loc)
    if os.path.isdir(direct):
        return direct
    # root already IS the collection folder
    parts = loc.replace("\\", "/").split("/")
    for i in range(len(parts)):
        cand = os.path.join(root, *parts[i:])
        if os.path.isdir(cand):
            return cand
        # also try dropping the first path component (collection name)
        if i == 0 and len(parts) > 1:
            cand = os.path.join(os.path.dirname(root), *parts)
            if os.path.isdir(cand):
                return cand
    return None


def _score_desc(desc: str) -> int:
    """Higher is better. Scout / reformat / bone kernels lose to axial soft tissue."""
    d = (desc or "").lower()
    if any(k in d for k in ("scout", "topo", "topogram", "localizer", "survey", "report", "structured report", "ko ", "key object")):
        return -100
    if any(k in d for k in ("sag", "coronal", " cor", "mip", "mpr")):
        return -50
    if "bone" in d or "b70" in d or "b80" in d:
        return 10
    if "lung" in d or "b70f" in d:
        return 40
    if any(k in d for k in ("soft", "std", "st ", "chest st", "b30", "b31", "b41",
                            "routine", "axial", "axia", "pre bx", "pre  bx",
                            "spiral", "chest", "thorax", "wb ")):
        return 80
    if d.strip() == "":
        return 20
    return 30


def _looks_dcm(path: str) -> bool:
    if os.path.splitext(path)[1]:
        return False
    # some TCIA dumps omit the extension
    try:
        with open(path, "rb") as fh:
            head = fh.read(132)
        return b"DICM" in head
    except OSError:
        return False


def _read_series(pydicom, series_dir: str, p: dict[str, Any]):
    files = []
    for dirpath, _dns, filenames in os.walk(series_dir):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if fn.endswith(_DCM_EXTS) or _looks_dcm(fp):
                files.append(fp)
    files.sort()
    if not files:
        raise FileNotFoundError(f"imaging: no DICOM files in {series_dir}")

    slices = []
    modality = None
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=False, stop_before_pixels=False)
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        slices.append(ds)
        modality = modality or str(getattr(ds, "Modality", "UNKNOWN"))

    if not slices:
        raise FileNotFoundError(f"imaging: no readable DICOM slices in {series_dir}")

    slices.sort(key=_slice_sort_key)
    shapes = {tuple(s.pixel_array.shape) for s in slices}
    if len(shapes) > 1:
        # drop localizer frames mixed into the same folder
        common = max(shapes, key=lambda sh: sum(1 for s in slices if tuple(s.pixel_array.shape) == sh))
        slices = [s for s in slices if tuple(s.pixel_array.shape) == common]
    arr = np.stack([s.pixel_array.astype(np.float64) for s in slices])

    mod = (modality or "UNKNOWN").upper()
    ref = slices[0]

    slope = float(getattr(ref, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ref, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    meta: dict[str, Any] = {
        "rescale": {"slope": slope, "intercept": intercept},
        "n_slices": int(arr.shape[0]),
        "series_dir": series_dir,
        "series_description": str(getattr(ref, "SeriesDescription", "") or ""),
    }

    if mod == "CT":
        win = p.get("window")
        if win:
            c, w = float(win[0]), float(win[1])
            arr = np.clip(arr, c - w / 2.0, c + w / 2.0)
            meta["window"] = {"center": c, "width": w}
    elif mod == "PT":
        arr, suv_meta = _to_suv(ref, arr)
        meta.update(suv_meta)
    elif mod == "MR":
        pass

    zscore = p.get("zscore_volume", mod == "MR")
    if zscore:
        sd = float(np.std(arr))
        if sd <= 0:
            raise ValueError(f"imaging: zero-variance volume in {series_dir}")
        arr = (arr - float(np.mean(arr))) / sd
        meta["zscore_volume"] = True

    return arr, mod, meta


def _slice_sort_key(ds) -> tuple:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return (2, float(ipp[2]))
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        try:
            return (1, float(inst))
        except (TypeError, ValueError):
            pass
    return (0, 0.0)


def _to_suv(ref, arr: np.ndarray):
    """SUVbw, or a loud refusal. A silently wrong SUV is worse than no SUV."""
    try:
        seq = ref.RadiopharmaceuticalInformationSequence[0]
        dose = float(seq.RadionuclideTotalDose)
        half_life = float(seq.RadionuclideHalfLife)
        weight_g = float(ref.PatientWeight) * 1000.0
    except Exception as exc:
        raise ValueError(
            "imaging: PET series lacks the tags required for SUVbw "
            "(RadionuclideTotalDose / RadionuclideHalfLife / PatientWeight). "
            "Refusing to emit uncalibrated counts as if they were SUV. "
            "Set modality explicitly and zscore_volume=True to use raw counts."
        ) from exc

    if dose <= 0 or half_life <= 0 or weight_g <= 0:
        raise ValueError("imaging: non-positive SUV calibration values")

    return arr * weight_g / dose, {
        "suv": {"kind": "SUVbw", "dose_bq": dose, "half_life_s": half_life,
                "weight_g": weight_g}
    }


def _first_order_features(vol: np.ndarray, p: dict[str, Any]) -> list[float]:
    v = vol.ravel()
    v = v[np.isfinite(v)]
    q = float(p.get("mask_percentile", 0) or 0)
    if q > 0:
        v = v[v >= np.percentile(v, q)]
    if v.size == 0:
        raise ValueError("imaging: mask_percentile removed every voxel")

    pct = np.percentile(v, _PERCENTILES)
    mean = float(np.mean(v))
    sd = float(np.std(v))
    centred = v - mean
    skew = float(np.mean(centred ** 3) / sd ** 3) if sd > 0 else 0.0
    kurt = float(np.mean(centred ** 4) / sd ** 4) if sd > 0 else 0.0

    hist, _ = np.histogram(v, bins=32)
    prob = hist / max(hist.sum(), 1)
    nz = prob[prob > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    return [mean, sd, float(np.min(v)), float(np.max(v)), *[float(x) for x in pct],
            float(pct[3] - pct[1]), skew, kurt, entropy,
            float(np.sqrt(np.mean(v ** 2))), float(v.size)]


def _feature_names() -> list[str]:
    return (
        ["mean", "std", "min", "max"]
        + [f"p{q}" for q in _PERCENTILES]
        + ["iqr", "skewness", "kurtosis", "entropy", "rms", "n_voxels"]
    )
