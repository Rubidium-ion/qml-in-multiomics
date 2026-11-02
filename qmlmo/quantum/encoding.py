"""Feature encoding + fidelity kernels, with mandatory degeneracy diagnostics.

The framework's one non-negotiable rule lives here: every quantum kernel run
reports `encoding_scale` and `kernel_offdiag_mean`. At scale=pi our 12-qubit
ZZFeatureMap collapsed to a mean off-diagonal mass of 9.7e-4 — numerically the
identity — while still producing publishable-looking AUCs near 0.75. Accuracy
alone conceals that failure completely.

Qiskit is optional. Without it, `FidelityKernel` falls back to an exact
NumPy statevector simulator for the same ZZ-feature-map family, so the whole
pipeline stays runnable in a bare environment.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def kernel_diagnostics(K: np.ndarray) -> dict[str, float]:
    """One-number degeneracy summary of a Gram matrix.

    offdiag_mean near 0  -> all pairs near-orthogonal, kernel ~ identity,
                            classifier cannot generalise regardless of biology
    offdiag_mean near 1  -> all pairs near-identical, kernel ~ all-ones,
                            no discriminative structure either
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"kernel must be square, got {K.shape}")
    n = K.shape[0]
    if n < 2:
        return {"offdiag_mean": 0.0, "offdiag_std": 0.0, "diag_mean": float(np.mean(np.diag(K))),
                "effective_rank": float(n), "degenerate": True}

    off = K[~np.eye(n, dtype=bool)]
    ev = np.linalg.eigvalsh((K + K.T) / 2.0)
    ev = np.clip(ev, 0.0, None)
    tot = float(ev.sum())
    if tot > 0:
        p = ev / tot
        p = p[p > 0]
        eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    else:
        eff_rank = 0.0

    om = float(np.mean(off))
    return {
        "offdiag_mean": om,
        "offdiag_std": float(np.std(off)),
        "diag_mean": float(np.mean(np.diag(K))),
        "effective_rank": eff_rank,
        # flags both failure directions, not just the concentration one
        "degenerate": bool(om < 0.01 or om > 0.99 or eff_rank < 2.0),
    }


class AngleEncoder:
    """Map features into rotation angles in [0, scale].

    `scale` is the hyperparameter this framework insists on reporting. It is
    fitted on training rows only; test rows are clipped into range so they
    cannot leave the encodable interval.
    """

    def __init__(self, scale: float = np.pi / 16, clip: bool = True) -> None:
        self.scale = float(scale)
        self.clip = clip
        self._lo: np.ndarray | None = None
        self._rng: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "AngleEncoder":
        X = np.asarray(X, dtype=np.float64)
        self._lo = np.nanmin(X, axis=0)
        rng = np.nanmax(X, axis=0) - self._lo
        rng[rng == 0] = 1.0
        self._rng = rng
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._lo is None or self._rng is None:
            raise RuntimeError("AngleEncoder.transform before fit")
        Z = (np.asarray(X, dtype=np.float64) - self._lo) / self._rng * self.scale
        return np.clip(Z, 0.0, self.scale) if self.clip else Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def describe(self) -> dict[str, Any]:
        return {"encoder": "angle", "encoding_scale": self.scale, "clip": self.clip}


def _zz_statevectors(Z: np.ndarray, reps: int = 1,
                     entanglement: str = "linear") -> np.ndarray:
    """Exact ZZFeatureMap statevectors, NumPy only.

    Diagonal feature map: |psi(x)> = U(x) H^{on} |0>, where U(x) is diagonal in
    the computational basis with phase
        sum_i x_i * b_i  +  sum_(i,j) in E (pi - x_i)(pi - x_j) * b_i * b_j
    matching Qiskit's ZZFeatureMap convention. Because U is diagonal, the whole
    map is a phase vector on the uniform superposition — no gate simulation
    needed, and it is exact.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, k = Z.shape
    dim = 1 << k

    idx = np.arange(dim)
    bits = ((idx[:, None] >> np.arange(k)[None, :]) & 1).astype(np.float64)  # (dim, k)

    if entanglement == "linear":
        pairs = [(i, i + 1) for i in range(k - 1)]
    elif entanglement == "full":
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    elif entanglement == "none":
        pairs = []
    else:
        raise ValueError(f"unknown entanglement {entanglement!r}")

    single = bits @ Z.T  # (dim, n)
    if pairs:
        pi = np.pi
        acc = np.zeros_like(single)
        for i, j in pairs:
            acc += np.outer(bits[:, i] * bits[:, j], np.ones(n)) * \
                   ((pi - Z[:, i]) * (pi - Z[:, j]))[None, :]
        phase = single + acc
    else:
        phase = single

    phase = phase * float(reps)
    amp = np.exp(1j * phase) / np.sqrt(dim)
    return amp.T  # (n, dim)


class FidelityKernel:
    """Exact fidelity kernel |<psi(x_i)|psi(x_j)>|^2 over a ZZ feature map."""

    def __init__(self, scale: float = np.pi / 16, reps: int = 1,
                 entanglement: str = "linear", max_qubits: int = 16) -> None:
        self.encoder = AngleEncoder(scale=scale)
        self.reps = int(reps)
        self.entanglement = entanglement
        self.max_qubits = int(max_qubits)
        self._train_states: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    def _states(self, X: np.ndarray, fit: bool) -> np.ndarray:
        k = X.shape[1]
        if k > self.max_qubits:
            raise ValueError(
                f"{k} features would need {k} qubits (2^{k} amplitudes). "
                f"max_qubits={self.max_qubits}. Reduce features inside the "
                "training fold — selecting them globally is a leakage bug."
            )
        Z = self.encoder.fit_transform(X) if fit else self.encoder.transform(X)
        return _zz_statevectors(Z, reps=self.reps, entanglement=self.entanglement)

    def fit(self, X: np.ndarray) -> "FidelityKernel":
        self._train_states = self._states(np.asarray(X, dtype=np.float64), fit=True)
        K = self._gram(self._train_states, self._train_states)
        self.diagnostics = {
            **self.encoder.describe(),
            **kernel_diagnostics(K),
            "reps": self.reps,
            "entanglement": self.entanglement,
            "n_qubits": int(X.shape[1]),
        }
        return self

    def train_kernel(self) -> np.ndarray:
        if self._train_states is None:
            raise RuntimeError("FidelityKernel.train_kernel before fit")
        return self._gram(self._train_states, self._train_states)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._train_states is None:
            raise RuntimeError("FidelityKernel.transform before fit")
        S = self._states(np.asarray(X, dtype=np.float64), fit=False)
        return self._gram(S, self._train_states)

    @staticmethod
    def _gram(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.abs(A @ B.conj().T) ** 2
