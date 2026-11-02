"""Modality adapter contract.

A FeatureBlock is the single currency of this framework. Every modality --
DICOM imaging, transcriptomics, genomics, metabolomics, synthetic -- reduces to
the same object, so integration never has to special-case a source.

Invariants enforced at construction (loudly, not silently):
  - X is 2-D float32/float64, shape (n_samples, n_features)
  - len(sample_ids) == n_samples, and ids are unique
  - len(feature_names) == n_features, and names are unique within the block
  - no NaN policy is applied here; cleaning is an explicit later stage
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class FeatureBlock:
    """One modality's feature matrix, aligned by sample_ids."""

    X: np.ndarray
    feature_names: list[str]
    sample_ids: list[str]
    modality: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        if self.X.ndim != 2:
            raise ValueError(
                f"[{self.modality}] X must be 2-D (n_samples, n_features), got {self.X.shape}"
            )
        n, k = self.X.shape
        self.sample_ids = [str(s) for s in self.sample_ids]
        self.feature_names = [str(f) for f in self.feature_names]

        if len(self.sample_ids) != n:
            raise ValueError(
                f"[{self.modality}] {len(self.sample_ids)} sample_ids for {n} rows"
            )
        if len(self.feature_names) != k:
            raise ValueError(
                f"[{self.modality}] {len(self.feature_names)} feature_names for {k} columns"
            )
        if len(set(self.sample_ids)) != n:
            dupes = _first_dupes(self.sample_ids)
            raise ValueError(f"[{self.modality}] duplicate sample_ids, e.g. {dupes}")
        if len(set(self.feature_names)) != k:
            dupes = _first_dupes(self.feature_names)
            raise ValueError(f"[{self.modality}] duplicate feature_names, e.g. {dupes}")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def missing_fraction(self) -> float:
        if self.X.size == 0:
            return 0.0
        return float(np.isnan(self.X).sum() / self.X.size)

    def subset_samples(self, ids: Sequence[str]) -> "FeatureBlock":
        """Reorder/restrict rows to `ids`. Missing ids are a hard error."""
        pos = {s: i for i, s in enumerate(self.sample_ids)}
        missing = [s for s in ids if s not in pos]
        if missing:
            raise KeyError(
                f"[{self.modality}] {len(missing)} requested sample(s) absent, "
                f"e.g. {missing[:3]}"
            )
        idx = np.asarray([pos[s] for s in ids], dtype=int)
        return FeatureBlock(
            X=self.X[idx],
            feature_names=list(self.feature_names),
            sample_ids=[str(s) for s in ids],
            modality=self.modality,
            meta={**self.meta, "subset_from_n": self.n_samples},
        )

    def prefixed_names(self) -> list[str]:
        """Namespaced feature names, so fused matrices stay traceable."""
        return [f"{self.modality}:{f}" for f in self.feature_names]

    def summary(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_fraction": round(self.missing_fraction, 6),
            "meta": self.meta,
        }


def _first_dupes(items: Sequence[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen and it not in out:
            out.append(it)
            if len(out) >= limit:
                break
        seen.add(it)
    return out


class ModalityAdapter(ABC):
    """Turns a raw source (directory, matrix file, generator) into a FeatureBlock."""

    #: short registry key, e.g. "imaging_dicom"
    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    @abstractmethod
    def load(self) -> FeatureBlock:
        """Produce the FeatureBlock. Must be deterministic given params."""

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, "params": self.params}
