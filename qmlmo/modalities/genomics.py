"""Genomics adapter: mutation / CNV / variant matrices.

Genomics blocks are usually sparse and often binary (mutated / not). Two
defaults matter:
  - `binarize` turns a VAF or count matrix into presence/absence, because a
    fidelity kernel on raw VAF mostly encodes sequencing depth, not biology
  - `min_prevalence` drops genes altered in almost nobody. With n in the low
    hundreds, a gene mutated in 2 samples cannot support a fold-level estimate,
    and keeping it inflates the feature count that the encoder has to spend
    qubits on.
"""
from __future__ import annotations

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter
from .matrix import MatrixAdapter


@register_modality
class GenomicsAdapter(ModalityAdapter):
    """params: path, feature_names, sample_ids, binarize=True, threshold=0.0,
    min_prevalence=0.02, modality='genomics'"""

    name = "genomics"

    def load(self) -> FeatureBlock:
        p = dict(self.params)
        modality = p.pop("modality", self.name)
        binarize = p.pop("binarize", True)
        threshold = float(p.pop("threshold", 0.0))
        min_prev = float(p.pop("min_prevalence", 0.02))

        block = MatrixAdapter(modality=modality, **p).load()
        X = block.X
        meta = dict(block.meta)

        if binarize:
            X = (np.nan_to_num(X, nan=0.0) > threshold).astype(np.float64)
            meta["binarize"] = {"threshold": threshold}

        names = list(block.feature_names)
        if min_prev > 0:
            prevalence = np.nanmean(X > 0, axis=0)
            keep = np.where(prevalence >= min_prev)[0]
            if keep.size == 0:
                raise ValueError(
                    f"{modality}: min_prevalence={min_prev} removed every feature "
                    f"(max prevalence {float(prevalence.max()):.4f}). Lower the "
                    "threshold rather than shipping an empty block."
                )
            dropped = X.shape[1] - keep.size
            X = X[:, keep]
            names = [names[i] for i in keep]
            meta["min_prevalence"] = min_prev
            meta["dropped_low_prevalence"] = int(dropped)

        return FeatureBlock(X=X, feature_names=names, sample_ids=block.sample_ids,
                            modality=modality, meta=meta)
