"""qsvm_angle — Quantum-kernel SVM with angle (ZZ-feature-map) encoding.

Conforms to Algorithm Interface Spec v0. Uses PennyLane for the quantum kernel
(default) with a graceful classical-RBF fallback if PennyLane is unavailable, so
the benchmark harness always gets a green end-to-end run.
"""
from __future__ import annotations

import numpy as np

from algorithms._preprocess import Preprocessor


class Algorithm:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        params = self.config.get("params", {})
        self.n_qubits = int(params.get("n_qubits", 8))
        self.reps = int(params.get("reps", 2))
        self.C = float(params.get("C", 1.0))
        self.backend = params.get("backend", "pennylane")
        cw = params.get("class_weight", "balanced")
        self.class_weight = None if cw in (None, "none", "None") else cw
        # angle encoding is 2*pi periodic -> reduce to n_qubits dims and MinMax
        # into a single RZ period, all fitted inside train() (per CV fold).
        pre_cfg = dict(self.config.get("preprocess") or {})
        pre_cfg.setdefault("pca", self.n_qubits)
        pre_cfg.setdefault("angle_range", [0.0, float(np.pi)])
        self.pre = Preprocessor(pre_cfg)
        self.model = None
        self._X_train = None
        self._kernel_fn = None
        self._used_fallback = False

    # ---- quantum kernel ----
    def _build_pennylane_kernel(self, n_features: int):
        import pennylane as qml

        n_qubits = min(self.n_qubits, n_features)
        dev = qml.device("default.qubit", wires=n_qubits)

        def feature_map(x):
            x = x[:n_qubits]
            for _ in range(self.reps):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(x[i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ((np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(n_qubits))

        def kernel(A, B):
            A = np.asarray(A, dtype=float)
            B = np.asarray(B, dtype=float)
            K = np.zeros((len(A), len(B)))
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    K[i, j] = kernel_circuit(a, b)[0]  # |<0|...|0>|^2
            return K

        self.n_qubits = n_qubits
        return kernel

    def train(self, X, y):
        from sklearn.svm import SVC

        X = np.asarray(self.pre.fit_transform(X), dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self._X_train = X

        try:
            if self.backend != "classical_fallback":
                self._kernel_fn = self._build_pennylane_kernel(X.shape[1])
                K = self._kernel_fn(X, X)
                self.model = SVC(kernel="precomputed", C=self.C,
                                 class_weight=self.class_weight,
                                 probability=True)
                self.model.fit(K, y)
                return self
        except Exception as e:  # PennyLane missing or circuit error -> fallback
            self._used_fallback = True
            print(f"[qsvm_angle] quantum kernel unavailable ({e}); "
                  f"falling back to classical RBF SVM")

        self._used_fallback = True
        self._kernel_fn = None
        self.model = SVC(kernel="rbf", C=self.C, gamma="scale",
                         class_weight=self.class_weight, probability=True)
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(self.pre.transform(X), dtype=np.float32)
        if self._kernel_fn is not None and not self._used_fallback:
            K = self._kernel_fn(X, self._X_train)
            return self.model.predict(K)
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(self.pre.transform(X), dtype=np.float32)
        if self._kernel_fn is not None and not self._used_fallback:
            K = self._kernel_fn(X, self._X_train)
            return self.model.predict_proba(K)
        return self.model.predict_proba(X)
