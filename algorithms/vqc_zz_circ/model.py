"""vqc_zz_circ — VQC arm with authors' 2026-08-08 A2A-specified configuration.

Delta vs `vqc_qiskit` (which is already ZZFeatureMap -> RealAmplitudes):
    reps_fm      1 -> 3
    reps_ansatz  2 -> 3
    entanglement default('full') -> 'circular'

Everything else (VarianceThreshold -> StandardScaler -> PCA(8) -> MinMax angle
scaling, COBYLA, seed) is inherited unchanged so the comparison is a clean
2-factor ablation and not a rewrite.
"""
from __future__ import annotations

from algorithms.vqc_qiskit.model import Algorithm as _Base


class Algorithm(_Base):
    pass
