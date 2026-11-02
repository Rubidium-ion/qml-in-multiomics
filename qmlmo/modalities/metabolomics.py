"""Metabolomics adapter: LC-MS / NMR intensity tables.

Metabolomics has two properties the omics pipeline must not paper over:
  - heavy right skew -> log transform after an explicit small offset
  - genuine missingness (below limit of detection), which is *informative*,
    not random. We record the missing pattern before imputing, and the default
    imputation is half-minimum per feature, the standard LOD convention.
"""
from __future__ import annotations

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter
from .matrix import MatrixAdapter


@register_modality
class MetabolomicsAdapter(ModalityAdapter):
    """params: path, feature_names, sample_ids, log=True, offset=1.0,
    impute='half_min'|'zero'|'none', max_missing=0.5, modality='metabolomics'"""

    name = "metabolomics"

    def load(self) -> FeatureBlock:
        p = dict(self.params)
        modality = p.pop("modality", self.name)
        do_log = p.pop("log", True)
        offset = float(p.pop("offset", 1.0))
        impute = p.pop("impute", "half_min")
        max_missing = float(p.pop("max_missing", 0.5))

        block = MatrixAdapter(modality=modality, **p).load()
        X = np.array(block.X, dtype=np.float64)
        names = list(block.feature_names)
        meta = dict(block.meta)

        miss = np.isnan(X).mean(axis=0)
        meta["missing_fraction_before"] = round(float(np.isnan(X).mean()), 6)

        if max_missing < 1.0:
            keep = np.where(miss <= max_missing)[0]
            if keep.size == 0:
                raise ValueError(
                    f"{modality}: every feature exceeds max_missing={max_missing}; "
                    f"min observed missingness is {float(miss.min()):.3f}"
                )
            meta["dropped_high_missing"] = int(X.shape[1] - keep.size)
            X = X[:, keep]
            names = [names[i] for i in keep]

        if impute == "half_min":
            for j in range(X.shape[1]):
                col = X[:, j]
                nan = np.isnan(col)
                if nan.any():
                    obs = col[~nan]
                    fill = (float(np.min(obs)) / 2.0) if obs.size else 0.0
                    col[nan] = fill
            meta["impute"] = "half_min"
        elif impute == "zero":
            X = np.nan_to_num(X, nan=0.0)
            meta["impute"] = "zero"
        elif impute != "none":
            raise ValueError(f"{modality}: unknown impute={impute!r}")

        if do_log:
            shifted = X + offset
            if np.nanmin(shifted) <= 0:
                raise ValueError(
                    f"{modality}: offset={offset} leaves non-positive values; "
                    "raise the offset or set log=False"
                )
            X = np.log(shifted)
            meta["log"] = {"offset": offset}

        return FeatureBlock(X=X, feature_names=names, sample_ids=block.sample_ids,
                            modality=modality, meta=meta)
