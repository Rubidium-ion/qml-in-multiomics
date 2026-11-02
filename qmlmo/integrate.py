"""Sample alignment, cleaning, normalisation and fusion of FeatureBlocks.

Two rules this module refuses to bend:

1. **Alignment is by explicit sample id, never by row position.** Position-based
   joins are the single most common silent bug in multi-omics integration.
2. **Every fitted transform is fitted on training rows only.** Normalisation
   fitted on the full matrix leaks test distribution into training and inflates
   exactly the marginal effects a quantum-advantage claim would rest on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .modalities.base import FeatureBlock


@dataclass
class FusedDataset:
    X: np.ndarray
    feature_names: list[str]
    sample_ids: list[str]
    block_slices: dict[str, tuple[int, int]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def block(self, modality: str) -> np.ndarray:
        a, b = self.block_slices[modality]
        return self.X[:, a:b]

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "blocks": {m: (b - a) for m, (a, b) in self.block_slices.items()},
            **self.meta,
        }


def intersect_samples(blocks: Sequence[FeatureBlock]) -> list[str]:
    """Ordered intersection of sample ids across blocks, with an honest report.

    Returns ids in the order of the first block, so results are deterministic.
    """
    if not blocks:
        raise ValueError("intersect_samples: no blocks given")
    common = set(blocks[0].sample_ids)
    for b in blocks[1:]:
        common &= set(b.sample_ids)
    ordered = [s for s in blocks[0].sample_ids if s in common]
    if not ordered:
        raise ValueError(
            "no samples shared across modalities. Ids per block: "
            + ", ".join(f"{b.modality}={b.n_samples}" for b in blocks)
            + ". Check that each adapter emits the same identifier space."
        )
    return ordered


def align(blocks: Sequence[FeatureBlock],
          sample_ids: Sequence[str] | None = None) -> tuple[list[FeatureBlock], dict]:
    """Restrict every block to a common, identically ordered sample set."""
    ids = list(sample_ids) if sample_ids is not None else intersect_samples(blocks)
    aligned = [b.subset_samples(ids) for b in blocks]
    report = {
        "n_aligned": len(ids),
        "per_block_before": {b.modality: b.n_samples for b in blocks},
        "dropped": {b.modality: b.n_samples - len(ids) for b in blocks},
    }
    return aligned, report


def clean_block(block: FeatureBlock, max_missing: float = 0.2,
                drop_zero_variance: bool = True) -> tuple[FeatureBlock, dict]:
    """Drop high-missing and zero-variance columns. Reports what it removed."""
    X = np.array(block.X, dtype=np.float64)
    names = list(block.feature_names)
    keep = np.arange(X.shape[1])

    miss = np.isnan(X).mean(axis=0)
    keep = keep[miss[keep] <= max_missing]
    dropped_missing = int(X.shape[1] - keep.size)

    dropped_var = 0
    if drop_zero_variance and keep.size:
        var = np.nanvar(X[:, keep], axis=0)
        before = keep.size
        keep = keep[var > 0]
        dropped_var = int(before - keep.size)

    if keep.size == 0:
        raise ValueError(
            f"[{block.modality}] cleaning removed every feature "
            f"({dropped_missing} high-missing, {dropped_var} zero-variance)"
        )

    out = FeatureBlock(
        X=X[:, keep],
        feature_names=[names[i] for i in keep],
        sample_ids=list(block.sample_ids),
        modality=block.modality,
        meta={**block.meta, "cleaned": True},
    )
    return out, {
        "modality": block.modality,
        "dropped_high_missing": dropped_missing,
        "dropped_zero_variance": dropped_var,
        "n_features_after": out.n_features,
    }


class BlockNormalizer:
    """Per-modality normalisation, fitted on training rows only.

    method:
      'zscore'  standardise each feature
      'minmax'  scale to [0, hi]; `hi` is the *encoding scale* for quantum arms
      'rank'    rank-inverse-normal, robust to omics outliers
      'none'
    """

    def __init__(self, method: str = "zscore", hi: float = 1.0,
                 clip: bool = True) -> None:
        if method not in {"zscore", "minmax", "rank", "none"}:
            raise ValueError(f"unknown normalisation method {method!r}")
        self.method = method
        self.hi = float(hi)
        self.clip = clip
        self._stats: dict[str, Any] = {}

    def fit(self, X: np.ndarray) -> "BlockNormalizer":
        X = np.asarray(X, dtype=np.float64)
        if self.method == "zscore":
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            sd[sd == 0] = 1.0
            self._stats = {"mu": mu, "sd": sd}
        elif self.method == "minmax":
            lo = np.nanmin(X, axis=0)
            hi = np.nanmax(X, axis=0)
            rng = hi - lo
            rng[rng == 0] = 1.0
            self._stats = {"lo": lo, "rng": rng}
        elif self.method == "rank":
            self._stats = {"ref": np.sort(X, axis=0)}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.method == "none":
            return X
        if not self._stats:
            raise RuntimeError("BlockNormalizer.transform called before fit")

        if self.method == "zscore":
            return (X - self._stats["mu"]) / self._stats["sd"]

        if self.method == "minmax":
            Z = (X - self._stats["lo"]) / self._stats["rng"] * self.hi
            return np.clip(Z, 0.0, self.hi) if self.clip else Z

        # rank-inverse-normal against the training reference distribution
        ref = self._stats["ref"]
        out = np.empty_like(X)
        n = ref.shape[0]
        for j in range(X.shape[1]):
            pos = np.searchsorted(ref[:, j], X[:, j], side="left")
            q = (pos + 0.5) / n
            out[:, j] = _probit(np.clip(q, 1e-6, 1 - 1e-6))
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _probit(q: np.ndarray) -> np.ndarray:
    """Inverse normal CDF without a scipy dependency (Acklam's rational approx)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)

    lo = q < plow
    if lo.any():
        t = np.sqrt(-2 * np.log(q[lo]))
        out[lo] = (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
                  ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    hi = q > phigh
    if hi.any():
        t = np.sqrt(-2 * np.log(1 - q[hi]))
        out[hi] = -(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
                   ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    mid = ~(lo | hi)
    if mid.any():
        t = q[mid] - 0.5
        r = t * t
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def fuse(blocks: Sequence[FeatureBlock]) -> FusedDataset:
    """Concatenate aligned blocks column-wise with namespaced feature names."""
    if not blocks:
        raise ValueError("fuse: no blocks")
    ids = blocks[0].sample_ids
    for b in blocks[1:]:
        if b.sample_ids != ids:
            raise ValueError(
                f"fuse: block {b.modality!r} is not aligned to {blocks[0].modality!r}. "
                "Call align() first — positional fusion of unaligned blocks is a "
                "silent correctness bug."
            )

    mats, names, slices, cursor = [], [], {}, 0
    for b in blocks:
        mats.append(b.X)
        names.extend(b.prefixed_names())
        slices[b.modality] = (cursor, cursor + b.n_features)
        cursor += b.n_features

    return FusedDataset(
        X=np.hstack(mats),
        feature_names=names,
        sample_ids=list(ids),
        block_slices=slices,
        meta={"blocks": [b.summary() for b in blocks]},
    )


def build_dataset(blocks: Iterable[FeatureBlock], max_missing: float = 0.2,
                  sample_ids: Sequence[str] | None = None) -> tuple[FusedDataset, dict]:
    """align -> clean -> fuse, with a provenance report at every step."""
    blocks = list(blocks)
    aligned, align_report = align(blocks, sample_ids)
    cleaned, clean_reports = [], []
    for b in aligned:
        cb, rep = clean_block(b, max_missing=max_missing)
        cleaned.append(cb)
        clean_reports.append(rep)
    fused = fuse(cleaned)
    report = {"align": align_report, "clean": clean_reports,
              "fused": {"n_samples": fused.n_samples, "n_features": fused.n_features}}
    fused.meta["provenance"] = report
    return fused, report
