"""_preprocess.py — in-train feature reduction shared by the classical/kernel algos.

Why this exists
---------------
`scripts/load_dataset.py --pca 0` now emits RAW features (log1p + top-variance
genes, ~5000 columns, no global scaler, no PCA). That is the only dataset layout
where k-fold CV is honest: nothing has seen the whole cohort.

The cost is that every algorithm must do its own scaling / reduction, and it must
do it **inside `train()`**, so each CV fold refits the transform on that fold's
training rows only. `vqc_qiskit` and `vqc_multiregister` already did this. This
module gives `svm_rbf`, `rf_baseline` and `qsvm_angle` the same discipline
instead of duplicating four slightly different pipelines.

Contract
--------
    pre = Preprocessor(config.get("preprocess"))
    Xt  = pre.fit_transform(X_train, )   # inside train()
    Xt  = pre.transform(X_test)          # inside predict()/predict_proba()

Config block (all optional)::

    preprocess:
      enabled: true          # false -> identity (use for pre-projected data)
      var_threshold: true    # drop zero-variance columns
      standardize: true
      pca: 8                 # int -> n_components; 0/null -> no PCA
      angle_range: null      # [lo, hi] -> MinMax into that range after PCA
      seed: 42

`enabled: auto` (the default) is the useful one: the transform switches itself
off when the incoming matrix already looks reduced (few columns relative to
rows), so pointing an algorithm at the legacy pre-projected `tcga_brca_v4`
dataset keeps the old numbers reproducible.
"""
from __future__ import annotations

import numpy as np

# a matrix with <= this many columns is assumed to be already reduced
AUTO_REDUCED_MAX_COLS = 64


class Preprocessor:
    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.mode = cfg.get("enabled", "auto")
        self.var_threshold = bool(cfg.get("var_threshold", True))
        self.standardize = bool(cfg.get("standardize", True))
        pca = cfg.get("pca", None)
        self.pca = int(pca) if pca else 0
        ar = cfg.get("angle_range", None)
        self.angle_range = tuple(float(v) for v in ar) if ar else None
        self.seed = int(cfg.get("seed", 42))
        self._pipe = None
        self._active = False

    # ------------------------------------------------------------------
    def _should_run(self, X) -> bool:
        if self.mode is False or self.mode == "false":
            return False
        if self.mode is True or self.mode == "true":
            return True
        # auto
        return X.shape[1] > AUTO_REDUCED_MAX_COLS

    def _build(self, n_features: int, n_samples: int):
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        steps = []
        if self.var_threshold:
            steps.append(("var", VarianceThreshold()))
        if self.standardize:
            steps.append(("std", StandardScaler()))
        if self.pca:
            # PCA cannot ask for more components than the fold provides
            n_comp = max(1, min(self.pca, n_features, n_samples - 1))
            steps.append(("pca", PCA(n_components=n_comp, random_state=self.seed)))
        if self.angle_range is not None:
            steps.append(("ang", MinMaxScaler(feature_range=self.angle_range)))
        return Pipeline(steps) if steps else None

    # ------------------------------------------------------------------
    def fit_transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self._active = self._should_run(X)
        if not self._active:
            self._pipe = None
            return X
        self._pipe = self._build(X.shape[1], X.shape[0])
        if self._pipe is None:
            return X
        return np.asarray(self._pipe.fit_transform(X), dtype=np.float32)

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self._active or self._pipe is None:
            return X
        Xt = np.asarray(self._pipe.transform(X), dtype=np.float32)
        if self.angle_range is not None:
            # test rows can land outside the train-fitted range; clip so no
            # sample silently wraps past the encoding period
            Xt = np.clip(Xt, self.angle_range[0], self.angle_range[1])
        return Xt

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "active": self._active,
            "steps": [n for n, _ in (self._pipe.steps if self._pipe else [])],
            "pca": self.pca,
            "angle_range": list(self.angle_range) if self.angle_range else None,
        }
