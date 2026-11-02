"""NIfTI volume adapter (OASIS-3 MR, and any .nii/.nii.gz cohort).

OASIS-3 ships MR as NIfTI, not DICOM, so the DICOM adapter cannot read it.
The important difference is not the container format but the calibration story:

  MR intensity is in arbitrary units. It is not comparable across scanners,
  sessions, or even sequences on the same scanner. Any pipeline that feeds raw
  MR intensity into a kernel is encoding acquisition settings, not biology.

So intensity normalisation here is MANDATORY, not a default that can be turned
off by accident. Two options, both standard in the MR literature:

  'zscore'   z-score within the brain mask (whole-volume statistics)
  'wm_mode'  scale so the white-matter intensity mode is 1.0 — more robust to
             lesion load, since a large lesion shifts the mean but not the mode

A crude Otsu-style foreground mask removes air, which otherwise dominates the
histogram and makes every subject look identical.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter

_PERCENTILES = (5, 25, 50, 75, 95)


def _require_nibabel():
    try:
        import nibabel  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "NIfTI support needs nibabel. Install with:\n"
            "    pip install nibabel\n"
            "(optional, so the omics-only path stays dependency-free)"
        ) from exc
    import nibabel

    return nibabel


@register_modality
class NiftiVolumeAdapter(ModalityAdapter):
    """params:
      root         : directory of .nii/.nii.gz, or of per-subject subdirectories
      pattern      : filename substring filter (e.g. 'T1w'), optional
      normalize    : 'zscore' | 'wm_mode' | 'none'  (default 'zscore')
      mask         : 'otsu' | 'none'  (default 'otsu')
      modality     : block label (default 'imaging_mr')
      subject_from : 'dirname' | 'filename'  (default auto)
    """

    name = "imaging_nifti"

    def load(self) -> FeatureBlock:
        nib = _require_nibabel()
        p = self.params
        root = p["root"]
        if not os.path.isdir(root):
            raise NotADirectoryError(f"nifti: {root} is not a directory")

        pattern = p.get("pattern")
        norm = p.get("normalize", "zscore")
        if norm not in {"zscore", "wm_mode", "none"}:
            raise ValueError(f"nifti: unknown normalize={norm!r}")
        if norm == "none" and not p.get("i_understand_mr_units_are_arbitrary"):
            raise ValueError(
                "nifti: normalize='none' on MR means feeding arbitrary units into "
                "the model, which encodes scanner settings rather than anatomy. "
                "Set i_understand_mr_units_are_arbitrary=true to override."
            )

        files = _collect(root, pattern)
        if not files:
            hint = f" matching pattern {pattern!r}" if pattern else ""
            raise FileNotFoundError(f"nifti: no .nii/.nii.gz under {root}{hint}")

        rows, ids = [], []
        for sid, fp in files:
            img = nib.load(fp)
            vol = np.asarray(img.dataobj, dtype=np.float64)
            vol = _normalise(vol, norm, p.get("mask", "otsu"))
            rows.append(_first_order(vol))
            ids.append(sid)

        return FeatureBlock(
            X=np.asarray(rows, dtype=np.float64),
            feature_names=_names(),
            sample_ids=ids,
            modality=p.get("modality", "imaging_mr"),
            meta={
                "adapter": self.name,
                "root": os.path.abspath(root),
                "n_subjects": len(ids),
                "normalize": norm,
                "mask": p.get("mask", "otsu"),
                "pattern": pattern,
                "note": "first-order intensity features; no texture, no registration",
            },
        )


def _collect(root: str, pattern: str | None) -> list[tuple[str, str]]:
    """Return (subject_id, path). One volume per subject; extras are an error."""
    found: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not (fn.endswith(".nii") or fn.endswith(".nii.gz")):
                continue
            if pattern and pattern not in fn:
                continue
            fp = os.path.join(dirpath, fn)
            sid = os.path.basename(dirpath) if dirpath != root else \
                fn.replace(".nii.gz", "").replace(".nii", "")
            if sid in found:
                raise ValueError(
                    f"nifti: subject {sid!r} has multiple matching volumes:\n"
                    f"  {found[sid]}\n  {fp}\n"
                    "Narrow `pattern` so exactly one sequence per subject is selected — "
                    "silently averaging T1 and T2 would be meaningless."
                )
            found[sid] = fp
    return sorted(found.items())


def _normalise(vol: np.ndarray, method: str, mask_mode: str) -> np.ndarray:
    v = vol
    if mask_mode == "otsu":
        m = _otsu_mask(v)
        if m.sum() < 100:
            raise ValueError("nifti: foreground mask kept <100 voxels; check the volume")
        v = np.where(m, v, np.nan)

    if method == "zscore":
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v))
        if sd <= 0:
            raise ValueError("nifti: zero-variance volume")
        return (v - mu) / sd

    if method == "wm_mode":
        fg = v[np.isfinite(v)]
        hist, edges = np.histogram(fg, bins=200)
        # white matter is the brightest dominant mode on T1; take the upper half
        upper = len(hist) // 2
        peak = upper + int(np.argmax(hist[upper:]))
        mode = float((edges[peak] + edges[peak + 1]) / 2.0)
        if mode == 0:
            raise ValueError("nifti: white-matter mode estimated as 0")
        return v / mode

    return v


def _otsu_mask(vol: np.ndarray) -> np.ndarray:
    """Two-class Otsu threshold. Separates head from air well enough for stats."""
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        raise ValueError("nifti: empty volume")
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        raise ValueError("nifti: empty histogram")
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    csum = np.cumsum(hist * centers)
    m0 = np.divide(csum, w0, out=np.zeros_like(csum), where=w0 > 0)
    m1 = np.divide(csum[-1] - csum, w1, out=np.zeros_like(csum), where=w1 > 0)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = -1
    return vol > centers[int(np.argmax(between))]


def _first_order(vol: np.ndarray) -> list[float]:
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        raise ValueError("nifti: no finite voxels after masking")
    pct = np.percentile(v, _PERCENTILES)
    mean = float(np.mean(v))
    sd = float(np.std(v))
    c = v - mean
    skew = float(np.mean(c ** 3) / sd ** 3) if sd > 0 else 0.0
    kurt = float(np.mean(c ** 4) / sd ** 4) if sd > 0 else 0.0
    hist, _ = np.histogram(v, bins=32)
    pr = hist / max(hist.sum(), 1)
    nz = pr[pr > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    return [mean, sd, float(v.min()), float(v.max()), *[float(x) for x in pct],
            float(pct[3] - pct[1]), skew, kurt, entropy,
            float(np.sqrt(np.mean(v ** 2))), float(v.size)]


def _names() -> list[str]:
    return (["mean", "std", "min", "max"] + [f"p{q}" for q in _PERCENTILES]
            + ["iqr", "skewness", "kurtosis", "entropy", "rms", "n_voxels"])
