"""qmlmo — multimodal QML experiment backend.

Design contract (v0):
  ModalityAdapter -> FeatureBlock(X, feature_names, sample_ids, modality, meta)
  Integrator      -> fuses aligned FeatureBlocks into a single design matrix
  Model           -> train/predict/predict_proba (same interface as algorithms/*)

Every run emits an encoding-scale + kernel off-diagonal diagnostic for quantum
arms. That is not optional decoration: the gate-refutation manuscript showed an
unreported encoding scale can swing fidelity-kernel AUC by 0.058, which exceeds
most published quantum-advantage effect sizes in this domain.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .modalities.base import FeatureBlock, ModalityAdapter  # noqa: F401
from .registry import get_modality, list_modalities, register_modality  # noqa: F401

__all__ = [
    "FeatureBlock",
    "ModalityAdapter",
    "get_modality",
    "list_modalities",
    "register_modality",
    "__version__",
]
