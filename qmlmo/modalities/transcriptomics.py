"""Transcriptomics adapter (RNA-seq TPM / counts).

Defaults encode what the GDSC/TCGA work already established:
  - log1p is the transform, applied once and recorded in meta
  - variance ranking alone is *wrong* for pathway biology, so a gene whitelist
    is force-rescued. PARP1 sat at variance rank 9,461 in the olaparib cohort:
    a top-5,000 variance filter deletes the drug target itself.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter
from .matrix import MatrixAdapter


@register_modality
class TranscriptomicsAdapter(ModalityAdapter):
    """params: path, feature_names, sample_ids, log1p=True, top_var_genes=None,
    whitelist=None (list or JSON path), modality='transcriptomics'"""

    name = "transcriptomics"

    def load(self) -> FeatureBlock:
        p = dict(self.params)
        modality = p.pop("modality", self.name)
        log1p = p.pop("log1p", True)
        top_var = p.pop("top_var_genes", None)
        whitelist = p.pop("whitelist", None)

        block = MatrixAdapter(modality=modality, **p).load()
        X = block.X
        meta = dict(block.meta)

        if log1p:
            if np.nanmin(X) < 0:
                raise ValueError(
                    f"{modality}: negative values present; log1p would produce NaN. "
                    "Pass log1p=False if this matrix is already transformed."
                )
            X = np.log1p(X)
            meta["log1p"] = True

        names = list(block.feature_names)
        if top_var is not None and top_var < len(names):
            keep = _variance_keep(X, names, int(top_var), _resolve_whitelist(whitelist))
            X = X[:, keep["idx"]]
            names = [names[i] for i in keep["idx"]]
            meta["top_var_genes"] = int(top_var)
            meta["whitelist_rescued"] = keep["rescued"]
            meta["whitelist_missing"] = keep["missing"]

        return FeatureBlock(X=X, feature_names=names, sample_ids=block.sample_ids,
                            modality=modality, meta=meta)


def _resolve_whitelist(w) -> list[str]:
    if w is None:
        return []
    if isinstance(w, str):
        if not os.path.exists(w):
            raise FileNotFoundError(
                f"whitelist {w} not found; refusing to fall back to variance-only "
                "selection silently"
            )
        with open(w) as fh:
            return [str(x) for x in json.load(fh)]
    return [str(x) for x in w]


def _variance_keep(X: np.ndarray, names: list[str], top_k: int,
                   whitelist: list[str]) -> dict:
    """Top-k by variance, unioned with whitelist genes that survive nowhere else."""
    var = np.nanvar(X, axis=0)
    order = np.argsort(var)[::-1]
    top = set(order[:top_k].tolist())

    pos = {n: i for i, n in enumerate(names)}
    rescued, missing = [], []
    for g in whitelist:
        i = pos.get(g)
        if i is None:
            missing.append(g)
        elif i not in top:
            top.add(i)
            rescued.append(g)

    return {"idx": sorted(top), "rescued": rescued, "missing": missing}
