"""Whole-slide pathology adapter (CAMELYON17, TCGA diagnostic slides).

WSI is not "a big image". Three properties break naive pipelines:

1. **Pyramidal and huge.** A CAMELYON17 slide is ~100k x 200k px at level 0.
   Nothing here ever reads level 0. We read a low-resolution level and compute
   slide-level statistics, which is honest about what is being measured.

2. **Mostly background.** Typically 70-90% of a slide is white glass. Without a
   tissue mask, every slide's histogram is dominated by background and all
   slides look identical. Tissue detection here is Otsu on the saturation
   channel, the standard cheap approach.

3. **Stain variation is a confounder, not noise.** H&E colour varies by scanner,
   protocol, and centre. CAMELYON17 is explicitly a *multi-centre* dataset --
   5 hospitals -- so stain differences correlate with centre, and centre may
   correlate with label. A model can score well by learning the scanner.
   `stain_normalize='macenko'` reduces this; the meta always records which
   centre-level confound control was applied, because reporting AUC without it
   on CAMELYON17 is close to meaningless.

Features are stain-decomposed density statistics plus tissue morphometry, not
raw RGB moments, so they carry some biological meaning.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter

_EXTS = (".tif", ".tiff", ".svs", ".ndpi")
_PERCENTILES = (10, 25, 50, 75, 90)

# Ruifrok & Johnston H&E stain vectors (optical density space)
_HE_REF = np.array([[0.5626, 0.2159],
                    [0.7201, 0.8012],
                    [0.4062, 0.5581]], dtype=np.float64)


def _require_tifffile():
    try:
        import tifffile  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "WSI support needs tifffile. Install with:\n"
            "    pip install tifffile\n"
            "For .svs/.ndpi vendor formats, openslide-python is preferable."
        ) from exc
    import tifffile

    return tifffile


@register_modality
class PathologyWSIAdapter(ModalityAdapter):
    """params:
      root             : directory of slides, or per-subject subdirectories
      level            : pyramid level to read (default -1 = smallest)
      max_pixels       : safety cap on decoded pixels (default 4e7)
      stain_normalize  : 'macenko' | 'none' (default 'macenko')
      tissue_threshold : saturation Otsu override, or None for automatic
      min_tissue_frac  : refuse slides with less tissue than this (default 0.01)
      modality         : block label (default 'pathology')
    """

    name = "pathology_wsi"

    def load(self) -> FeatureBlock:
        tiff = _require_tifffile()
        p = self.params
        root = p["root"]
        if not os.path.isdir(root):
            raise NotADirectoryError(f"wsi: {root} is not a directory")

        slides = _collect(root)
        if not slides:
            raise FileNotFoundError(
                f"wsi: no slides ({', '.join(_EXTS)}) under {root}"
            )

        stain_norm = p.get("stain_normalize", "macenko")
        if stain_norm not in {"macenko", "none"}:
            raise ValueError(f"wsi: unknown stain_normalize={stain_norm!r}")

        rows, ids, skipped = [], [], []
        for sid, fp in slides:
            rgb = _read_level(tiff, fp, p.get("level", -1),
                              float(p.get("max_pixels", 4e7)))
            mask = _tissue_mask(rgb, p.get("tissue_threshold"))
            frac = float(mask.mean())
            if frac < float(p.get("min_tissue_frac", 0.01)):
                skipped.append((sid, round(frac, 5)))
                continue
            rows.append(_slide_features(rgb, mask, stain_norm))
            ids.append(sid)

        if not rows:
            raise ValueError(
                f"wsi: every slide fell below min_tissue_frac; "
                f"tissue fractions: {skipped[:5]}"
            )

        return FeatureBlock(
            X=np.asarray(rows, dtype=np.float64),
            feature_names=_names(),
            sample_ids=ids,
            modality=p.get("modality", "pathology"),
            meta={
                "adapter": self.name,
                "root": os.path.abspath(root),
                "n_slides": len(ids),
                "level": p.get("level", -1),
                "stain_normalize": stain_norm,
                "skipped_low_tissue": skipped,
                "confound_warning": (
                    "multi-centre WSI cohorts (e.g. CAMELYON17, 5 hospitals) have "
                    "stain/scanner variation correlated with centre. Report "
                    "leave-one-centre-out performance, not only random-split AUC."
                ),
                "note": "slide-level first-order + stain-density features; no tiling, no MIL",
            },
        )


def _collect(root: str) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for dirpath, _dn, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(_EXTS):
                continue
            fp = os.path.join(dirpath, fn)
            sid = os.path.splitext(fn)[0]
            if sid in found:
                raise ValueError(f"wsi: duplicate slide id {sid!r}: {found[sid]} and {fp}")
            found[sid] = fp
    return sorted(found.items())


def _read_level(tiff, path: str, level: int, max_pixels: float) -> np.ndarray:
    """Read one pyramid level as uint8 RGB, smallest-first fallback."""
    with tiff.TiffFile(path) as tf:
        series = tf.series[0]
        levels = getattr(series, "levels", None) or [series]
        idx = level if level >= 0 else len(levels) + level
        idx = max(0, min(idx, len(levels) - 1))

        # step down until under the pixel cap rather than OOM on a huge level
        while idx < len(levels) - 1:
            shp = levels[idx].shape
            if int(np.prod(shp[:2])) <= max_pixels:
                break
            idx += 1

        arr = levels[idx].asarray()

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[0] in (3, 4) and arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3]

    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        hi = float(a.max()) or 1.0
        arr = np.clip(a / hi * 255.0, 0, 255).astype(np.uint8)
    return arr


def _tissue_mask(rgb: np.ndarray, threshold: float | None) -> np.ndarray:
    """Otsu on saturation: glass is white (low saturation), tissue is stained."""
    a = rgb.astype(np.float64) / 255.0
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)

    t = float(threshold) if threshold is not None else _otsu(sat)
    # also drop near-black scan artefacts
    return (sat > t) & (mx > 0.05)


def _otsu(v: np.ndarray) -> float:
    hist, edges = np.histogram(v.ravel(), bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.1
    w0 = np.cumsum(hist)
    w1 = total - w0
    csum = np.cumsum(hist * centers)
    m0 = np.divide(csum, w0, out=np.zeros_like(csum), where=w0 > 0)
    m1 = np.divide(csum[-1] - csum, w1, out=np.zeros_like(csum), where=w1 > 0)
    between = w0 * w1 * (m0 - m1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1
    return float(centers[int(np.argmax(between))])


def _od(rgb: np.ndarray) -> np.ndarray:
    """Optical density; Beer-Lambert."""
    a = np.clip(rgb.astype(np.float64), 1.0, 255.0) / 255.0
    return -np.log(a)


def _macenko_stains(od_flat: np.ndarray) -> np.ndarray:
    """Estimate 2 stain vectors via the Macenko SVD-angle method.

    Falls back to reference H&E vectors when the slide has too little stain
    variation to estimate them — a silent bad estimate is worse than a
    documented fallback.
    """
    keep = od_flat[(od_flat > 0.15).any(axis=1)]
    if keep.shape[0] < 100:
        return _HE_REF
    try:
        _u, _s, vt = np.linalg.svd(keep - keep.mean(axis=0), full_matrices=False)
    except np.linalg.LinAlgError:
        return _HE_REF
    plane = vt[:2].T
    proj = keep @ plane
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(ang, 1), np.percentile(ang, 99)
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    M = np.stack([np.abs(v1), np.abs(v2)], axis=1)
    n = np.linalg.norm(M, axis=0)
    n[n == 0] = 1.0
    return M / n


def _slide_features(rgb: np.ndarray, mask: np.ndarray, stain_norm: str) -> list[float]:
    od = _od(rgb)
    od_t = od[mask]                       # (n_tissue, 3)
    if od_t.shape[0] < 50:
        raise ValueError("wsi: fewer than 50 tissue pixels after masking")

    stains = _macenko_stains(od_t) if stain_norm == "macenko" else _HE_REF
    conc = np.linalg.lstsq(stains, od_t.T, rcond=None)[0]   # (2, n_tissue)
    h, e = conc[0], conc[1]

    feats: list[float] = []
    for c in (h, e):
        pct = np.percentile(c, _PERCENTILES)
        feats += [float(np.mean(c)), float(np.std(c)), *[float(x) for x in pct]]

    # H/E ratio: nuclear-to-stromal proxy, the biologically meaningful summary
    em = float(np.mean(e))
    feats.append(float(np.mean(h) / em) if em > 1e-6 else 0.0)

    # tissue morphometry
    feats.append(float(mask.mean()))
    feats.append(float(mask.sum()))

    # greyscale texture proxy: entropy + gradient energy over tissue
    grey = rgb.astype(np.float64).mean(axis=-1)
    gt = grey[mask]
    hist, _ = np.histogram(gt, bins=32)
    pr = hist / max(hist.sum(), 1)
    nz = pr[pr > 0]
    feats.append(float(-(nz * np.log2(nz)).sum()))

    gy, gx = np.gradient(grey)
    gmag = np.sqrt(gx ** 2 + gy ** 2)[mask]
    feats += [float(np.mean(gmag)), float(np.std(gmag))]
    return feats


def _names() -> list[str]:
    out: list[str] = []
    for s in ("hematoxylin", "eosin"):
        out += [f"{s}_mean", f"{s}_std"] + [f"{s}_p{q}" for q in _PERCENTILES]
    out += ["he_ratio", "tissue_fraction", "tissue_pixels",
            "grey_entropy", "grad_mean", "grad_std"]
    return out
