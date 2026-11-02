"""rf_baseline — RandomForest baseline (Algorithm Interface Spec v0).

RF does not need scaling, but it must still not be handed a globally-PCA'd
matrix during k-fold CV. The Preprocessor defaults to variance filtering only
(no PCA) so the forest sees the raw gene space, refit per fold.
"""
from __future__ import annotations
import numpy as np

from algorithms._preprocess import Preprocessor


class Algorithm:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        p = self.config.get("params", {})
        self.n_estimators = int(p.get("n_estimators", 100))
        self.seed = int(p.get("seed", 42))
        pre_cfg = dict(self.config.get("preprocess") or {})
        pre_cfg.setdefault("standardize", False)  # trees are scale-invariant
        pre_cfg.setdefault("seed", self.seed)
        self.pre = Preprocessor(pre_cfg)
        self.model = None

    def train(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        Xt = self.pre.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed
        )
        self.model.fit(Xt, np.asarray(y, np.int32))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self.pre.transform(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self.pre.transform(X))
