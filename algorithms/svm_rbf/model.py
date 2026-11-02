"""svm_rbf — classical RBF-kernel SVM baseline (Algorithm Interface Spec v0).

Preprocessing (VarianceThreshold -> StandardScaler -> optional PCA) happens
inside train(), so a k-fold harness refits it per fold and the score is not
inflated by a transform that saw the test rows. See algorithms/_preprocess.py.
"""
from __future__ import annotations
import numpy as np

from algorithms._preprocess import Preprocessor


class Algorithm:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        p = self.config.get("params", {})
        self.C = float(p.get("C", 1.0))
        self.gamma = p.get("gamma", "scale")
        self.pre = Preprocessor(self.config.get("preprocess"))
        self.model = None

    def train(self, X, y):
        from sklearn.svm import SVC
        Xt = self.pre.fit_transform(X)
        self.model = SVC(C=self.C, gamma=self.gamma, kernel="rbf", probability=True)
        self.model.fit(Xt, np.asarray(y, np.int32))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self.pre.transform(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self.pre.transform(X))
