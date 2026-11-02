"""Pathology whole-slide image (WSI) adapter — CAMELYON-class cohorts.

Scope, stated honestly: this is *not* a real histology feature extractor. There
is no stain normalisation, no tissue/nucleus segmentation, and no deep patch
features. What it guarantees is a deterministic, benchmark-grade tile-statistics
block so quantum-vs-classical comparisons can run on pathology cohorts without
a multi-GB whole-slide download for every sample.

Two input modes, auto-detected per slide directory:

  tiles  (pure NumPy, no extra dependency)
      <root>/<slide>/tile_*.npy      2-D intensity tiles (any scale)
      Deterministic grid = sorted tile filenames. Used for reproducible
      smoke tests and for pre-tiled / thumbnail cohorts.

  tiff   (whole slide; needs `tifffile`)
      <root>/<slide>/<anything>.tif|.svs
      Reads the highest pyramid level whose width <= analysis_max_dim
      (default 2048), then cuts a fixed non-overlapping tile grid over that
      analysis image. CAMELYON17 level-0 WSIs are ~200k px wide; this never
      touches full resolution.

All features are computed on tiles z-scored by the *analysis-image* global
mean/std, which makes them scale-invariant across raw .npy values and 0-255
TIFF luminance. First-order statistics + two texture proxies (gradient energy,
flat-tile fraction). No GLCM/GLRLM yet — same honest limitation as the DICOM
adapter, stated once in the docstring rather than silently.

Sample ids are slide directory names; labels (e.g. CAMELYON17 metastasis
presence) are supplied separately by the config `labels:` section keyed on
those ids.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter

_TILE_EXT = (".npy",)
_SLIDE_EXT = (".tif", ".tiff", ".svs")
_TILE_STAT_KEYS = ("tile_mean", "tile_std", "tile_grad")


def _require_tifffile():
    try:
        import tifffile  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "whole-slide TIFF support needs tifffile. Install with:\n"
            "    pip install tifffile\n"
            "(kept optional so the tile mode has zero imaging dependencies)"
        ) from exc
    return tifffile


def _lum(a: np.ndarray) -> np.ndarray:
    """RGB(A) -> luminance in [0, 255]-equivalent space; pass grayscale through."""
    if a.ndim == 3:
        rgb = a[..., :3].astype(np.float64)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return a.astype(np.float64)


def _tile_grid(img: np.ndarray, tile: int) -> list[np.ndarray]:
    h, w = img.shape
    out = []
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            out.append(img[y0 : y0 + tile, x0 : x0 + tile])
    return out


def _slide_from_tiff(tifffile, path: str, max_dim: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tf:
        if not tf.series:
            raise ValueError(f"pathology: no image series in {path}")
        series = tf.series[0]
        levels = series.levels
        # pick the smallest level whose width fits analysis_max_dim
        for lvl in levels:
            shape = lvl.shape  # (h, w[, c])
            if shape[1] <= max_dim:
                return _lum(lvl.asarray())
        return _lum(levels[-1].asarray())


def _analysis_image(tifffile, path: str, max_dim: int, tile: int) -> tuple[np.ndarray, dict]:
    img = _slide_from_tiff(tifffile, path, max_dim)
    return img, {"analysis_h": int(img.shape[0]), "analysis_w": int(img.shape[1]),
                 "analysis_tile": tile}


def _npy_tiles(tile_dir: str) -> tuple[list[np.ndarray], dict]:
    files = sorted(f for f in os.listdir(tile_dir) if f.endswith(_TILE_EXT))
    if not files:
        raise FileNotFoundError(
            f"pathology: no *.npy tiles under {tile_dir} "
            "(expected tile_*.npy per slide; .tif/.svs files are also accepted)"
        )
    tiles = [np.asarray(np.load(os.path.join(tile_dir, f))) for f in files]
    return tiles, {"n_tile_files": len(files)}


def _features_from_tiles(tiles: list[np.ndarray]) -> dict[str, float]:
    """Scale-invariant tile statistics over a whole slide (or tile set)."""
    stacked = np.concatenate([t.ravel() for t in tiles])
    gmean, gstd = float(stacked.mean()), float(stacked.std())
    if gstd <= 0:
        gstd = 1.0  # constant slide; z-scoring disabled, all features degenerate
    tiles = [t.astype(np.float64) for t in tiles]
    z = [(t - gmean) / gstd for t in tiles]
    # per-tile summary stats
    tm = np.asarray([float(t.mean()) for t in z])          # tile means
    ts = np.asarray([float(t.std()) for t in z])           # tile stds
    tg = np.asarray([float(np.abs(np.gradient(t)).mean()) for t in z])  # gradient energy
    hist, _ = np.histogram(stacked, bins=64, density=True)
    p = hist[hist > 0]
    entropy = float(-np.sum(p * np.log(p)))
    flat_frac = float(np.mean(ts < 0.15))  # std well below the global std -> background

    feat: dict[str, float] = {"intensity_hist_entropy": entropy,
                              "flat_tile_fraction": flat_frac}
    for key, vals in zip(_TILE_STAT_KEYS, (tm, ts, tg)):
        feat[f"{key}_mean"] = float(np.mean(vals))
        feat[f"{key}_std"] = float(np.std(vals))
        for q, lab in ((0.05, "p5"), (0.5, "p50"), (0.95, "p95")):
            feat[f"{key}_{lab}"] = float(np.quantile(vals, q))
    feat["n_tiles"] = float(len(tiles))
    return feat


@register_modality
class PathologyWSIAdapter(ModalityAdapter):
    """params:
      root             : directory holding one subdirectory per slide
      analysis_max_dim : tiff mode only — highest pyramid level with width <= this
                         (default 2048)
      analysis_tile    : tile edge in px at the analysis level (default 256)
    """

    name = "pathology_wsi"

    def load(self) -> FeatureBlock:
        p = self.params
        root = p["root"]
        if not os.path.isdir(root):
            raise NotADirectoryError(f"pathology: {root} is not a directory")

        slides = sorted(d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d)))
        if not slides:
            raise FileNotFoundError(
                f"pathology: no slide subdirectories under {root}. Expected "
                "one directory per slide, each holding *.npy tiles or one *.tif/.svs."
            )

        max_dim = int(p.get("analysis_max_dim", 2048))
        tile = int(p.get("analysis_tile", 256))
        tifffile = None  # lazily imported only if a slide needs tiff mode

        rows: list[dict[str, float]] = []
        used_tiff = False
        slide_modes: list[str] = []
        for sid in slides:
            d = os.path.join(root, sid)
            entries = [f for f in os.listdir(d) if not f.startswith(".")]
            tiffs = [f for f in entries if f.lower().endswith(_SLIDE_EXT)]
            npy = [f for f in entries if f.lower().endswith(_TILE_EXT)]
            if npy:
                tiles, tmeta = _npy_tiles(d)
                slide_modes.append("tiles")
            elif tiffs:
                if tifffile is None:
                    tifffile = _require_tifffile()
                img, imeta = _analysis_image(
                    tifffile, os.path.join(d, sorted(tiffs)[0]), max_dim, tile)
                tiles = _tile_grid(img, tile)
                slide_modes.append("tiff")
                used_tiff = True
            else:
                raise FileNotFoundError(
                    f"pathology: slide {sid!r} has no *.npy tiles and no *.tif/.svs "
                    f"file (found: {entries[:5]})"
                )
            r = _features_from_tiles(tiles)
            r["n_flat_frac_ok"] = float(r["flat_tile_fraction"] < 0.9)
            rows.append(r)
            if len(slides) == 1:
                rows[-1].update(tmeta if npy else imeta)

        feature_names = sorted(rows[0].keys())
        X = np.asarray([[r[k] for k in feature_names] for r in rows], dtype=np.float64)
        return FeatureBlock(
            X=X,
            feature_names=feature_names,
            sample_ids=slides,
            modality="pathology",
            meta={
                "adapter": self.name,
                "root": os.path.abspath(root),
                "slide_modes": slide_modes,
                "used_tiff": used_tiff,
                "analysis_max_dim": max_dim,
                "analysis_tile": tile,
                "n_slides": len(slides),
                "note": "z-scored tile statistics only; no stain normalisation, "
                        "no GLCM/GLRLM, no deep patch features",
            },
        )
