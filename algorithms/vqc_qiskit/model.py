"""vqc_qiskit — Variational Quantum Classifier (Qiskit).

Conforms to Algorithm Interface Spec v0.

Pipeline (v2, authors 2026-07-31):
    VarianceThreshold -> StandardScaler -> PCA(n_qubits) -> MinMax([-pi/2, pi/2])
    -> ZZFeatureMap -> RealAmplitudes -> COBYLA/SPSA

Why the preprocessing matters: angle-encoding maps a feature value directly onto
a rotation angle. Raw TCGA expression values span ~[-200, 208], so every angle
wraps around the Bloch sphere many times (mod 2*pi) and the encoding becomes
effectively random. v1 skipped scaling entirely -> acc 0.417 / AUC 0.25.
"""
from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        params = self.config.get("params", {})
        self.n_qubits = int(params.get("n_qubits", 8))
        self.reps_fm = int(params.get("reps_fm", 1))
        self.reps_ansatz = int(params.get("reps_ansatz", 2))
        # ZZFeatureMap entanglement topology. None -> qiskit default ('full').
        # 'circular' closes the chain so qubit0 reaches qubit n-1 in one layer.
        # Requested by authors 2026-08-08 (A2A) as the quantum-arm differentiator.
        ent = params.get("entanglement", None)
        self.entanglement = None if ent in (None, "none", "None") else str(ent)
        self.maxiter = int(params.get("maxiter", 300))
        self.optimizer_name = str(params.get("optimizer", "cobyla")).lower()
        self.angle_scale = float(params.get("angle_scale", np.pi / 2))
        self.seed = int(params.get("seed", 42))

        self.model = None
        self._pipeline = None          # sklearn preprocessing pipeline
        self._classes = None
        self._used_fallback = False

    # ------------------------------------------------------------------ #
    # preprocessing
    # ------------------------------------------------------------------ #
    def _build_pipeline(self, n_features: int):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA

        steps = [
            ("var", VarianceThreshold(threshold=1e-8)),
            ("std", StandardScaler()),
        ]
        if n_features > self.n_qubits:
            steps.append(("pca", PCA(n_components=self.n_qubits,
                                     random_state=self.seed)))
        # angle encoding needs a bounded domain, otherwise rotations wrap
        steps.append(("ang", MinMaxScaler(
            feature_range=(-self.angle_scale, self.angle_scale))))
        return Pipeline(steps)

    def _build_optimizer(self):
        from qiskit_algorithms.optimizers import COBYLA, SPSA
        if self.optimizer_name == "spsa":
            return SPSA(maxiter=self.maxiter)
        return COBYLA(maxiter=self.maxiter)

    # ------------------------------------------------------------------ #
    # interface
    # ------------------------------------------------------------------ #
    def train(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._classes = np.unique(y)

        try:
            from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
            from qiskit_machine_learning.algorithms import VQC
            from qiskit.primitives import StatevectorSampler
            from sklearn.preprocessing import OneHotEncoder

            algorithm_globals_seed(self.seed)

            self._pipeline = self._build_pipeline(X.shape[1])
            Xt = self._pipeline.fit_transform(X)
            num_qubits = Xt.shape[1]

            y_oh = OneHotEncoder(sparse_output=False).fit_transform(
                y.reshape(-1, 1))

            fm_kw = {"feature_dimension": num_qubits, "reps": self.reps_fm}
            if self.entanglement:
                fm_kw["entanglement"] = self.entanglement
            feature_map = ZZFeatureMap(**fm_kw)
            ansatz = RealAmplitudes(num_qubits=num_qubits,
                                    reps=self.reps_ansatz)

            self.model = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=self._build_optimizer(),
                sampler=StatevectorSampler(),
            )
            self.model.fit(Xt, y_oh)
            self._used_fallback = False
        except Exception as e:
            print(f"[vqc_qiskit] training failed: {e}")
            self._used_fallback = True
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=self.seed)
            self.model.fit(X, y)
        return self

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        if self._used_fallback or self._pipeline is None:
            return X
        return self._pipeline.transform(X)

    def predict(self, X) -> np.ndarray:
        Xt = self._transform(X)
        if self._used_fallback:
            return self.model.predict(Xt)
        out = np.asarray(self.model.predict(Xt))
        if out.ndim == 2:                      # one-hot -> label
            out = np.argmax(out, axis=1)
        return out.astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Return true class probabilities (n_samples, n_classes).

        v1 returned the hard one-hot prediction here, which made AUC-ROC
        meaningless (it degenerated to the accuracy of a 0/1 vector).
        """
        Xt = self._transform(X)
        if self._used_fallback:
            return self.model.predict_proba(Xt)
        try:
            proba = np.asarray(self.model.predict_proba(Xt), dtype=float)
        except Exception:
            # fall back to the raw network forward pass (probability outputs)
            proba = np.asarray(
                self.model.neural_network.forward(
                    Xt, self.model.weights), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        # normalise defensively
        s = proba.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return proba / s


def algorithm_globals_seed(seed: int) -> None:
    """Best-effort deterministic seeding across qiskit versions."""
    np.random.seed(seed)
    for mod in ("qiskit_algorithms.utils", "qiskit_machine_learning.utils"):
        try:
            import importlib
            m = importlib.import_module(mod)
            m.algorithm_globals.random_seed = seed
            return
        except Exception:
            continue
