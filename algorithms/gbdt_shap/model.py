"""gbdt_shap — gradient-boosted trees arm for the interaction-capture comparison.

Why this arm exists
-------------------
bio's question (relayed by authors, 2026-08-08): the QML arms lost to SVM-RBF, but
RBF is a *similarity* kernel -- it never names an interaction. If the hoped-for
quantum advantage is "efficiently capturing feature interactions", then the fair
classical opponent is a model that builds interactions **explicitly**: a GBDT,
where every path of depth d encodes a d-way interaction.

Discipline
----------
Same contract as every other arm: all fitting happens inside `train()`, so under
k-fold CV each fold refits the variance filter and the booster on that fold's
training rows only. Nothing sees the whole cohort.

The top_k variance pre-filter is NOT a performance trick -- it exists because
SHAP interaction values are O(n_features^2) and 5000^2 is intractable. It is
fitted per-fold like everything else.
"""
from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        p = self.config.get("params", {})
        self.n_estimators = int(p.get("n_estimators", 300))
        self.learning_rate = float(p.get("learning_rate", 0.05))
        self.max_depth = int(p.get("max_depth", 3))
        self.subsample = float(p.get("subsample", 0.8))
        self.top_k = int(p.get("top_k", 300))
        self.seed = int(p.get("seed", 42))
        self.model = None
        self._keep = None          # variance-filter column index, per fold

    def _fit_filter(self, X):
        v = np.var(np.asarray(X, dtype=float), axis=0)
        k = min(self.top_k, X.shape[1])
        self._keep = np.argsort(v)[::-1][:k]
        return self._keep

    def train(self, X, y):
        from sklearn.ensemble import GradientBoostingClassifier

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._fit_filter(X)
        Xt = X[:, self._keep]

        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.seed,
        )
        self.model.fit(Xt, y)
        return self

    def _tx(self, X):
        return np.asarray(X, dtype=float)[:, self._keep]

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._tx(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._tx(X))
