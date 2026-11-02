"""Model zoo: classical baselines + quantum kernel arms, one interface.

Every model exposes fit / predict / predict_proba and a `.report` dict. Quantum
arms put their encoding scale and kernel diagnostics into `.report`, which the
runner copies into the result JSON. That is how the reporting standard proposed
in the gate-refutation paper becomes structurally unavoidable here rather than a
convention people forget.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .quantum.encoding import FidelityKernel

_REGISTRY: dict[str, type] = {}


def register_model(cls):
    _REGISTRY[cls.name] = cls
    return cls


def build_model(name: str, **params):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def list_models() -> list[str]:
    return sorted(_REGISTRY)


class BaseModel:
    name = "base"
    quantum = False

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.report: dict[str, Any] = {}

    def fit(self, X, y):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, X):
        return None


@register_model
class SVMRBF(BaseModel):
    name = "svm_rbf"

    def fit(self, X, y):
        from sklearn.svm import SVC
        self._m = SVC(C=self.params.get("C", 1.0),
                      gamma=self.params.get("gamma", "scale"),
                      probability=True,
                      random_state=self.params.get("seed", 42))
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


@register_model
class GBDT(BaseModel):
    name = "gbdt"

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingClassifier
        self._m = GradientBoostingClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", 3),
            learning_rate=self.params.get("learning_rate", 0.1),
            random_state=self.params.get("seed", 42))
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


@register_model
class RandomForest(BaseModel):
    name = "rf"

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self._m = RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            random_state=self.params.get("seed", 42), n_jobs=-1)
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


@register_model
class LogReg(BaseModel):
    name = "logreg"

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self._m = LogisticRegression(max_iter=2000, C=self.params.get("C", 1.0))
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


@register_model
class QSVMFidelity(BaseModel):
    """Quantum fidelity-kernel SVM with mandatory degeneracy reporting."""

    name = "qsvm_fidelity"
    quantum = True

    def fit(self, X, y):
        from sklearn.svm import SVC
        self._k = FidelityKernel(
            scale=self.params.get("encoding_scale", np.pi / 16),
            reps=self.params.get("reps", 1),
            entanglement=self.params.get("entanglement", "linear"),
            max_qubits=self.params.get("max_qubits", 14))
        self._k.fit(X)
        K = self._k.train_kernel()
        self._m = SVC(kernel="precomputed", C=self.params.get("C", 1.0),
                      probability=True, random_state=self.params.get("seed", 42))
        self._m.fit(K, y)
        self.report = dict(self._k.diagnostics)
        if self.report.get("degenerate"):
            self.report["warning"] = (
                "kernel is degenerate: off-diagonal mass "
                f"{self.report['offdiag_mean']:.3g}. Accuracy from this arm is "
                "not evidence about quantum structure — retune encoding_scale."
            )
        return self

    def predict(self, X):
        return self._m.predict(self._k.transform(X))

    def predict_proba(self, X):
        return self._m.predict_proba(self._k.transform(X))
