"""Modality adapters. Importing this package registers every built-in adapter."""
from __future__ import annotations

from .base import FeatureBlock, ModalityAdapter  # noqa: F401
from .genomics import GenomicsAdapter  # noqa: F401
from .imaging_dicom import DicomRadiomicsAdapter  # noqa: F401
from .imaging_nifti import NiftiVolumeAdapter  # noqa: F401
from .matrix import MatrixAdapter  # noqa: F401
from .metabolomics import MetabolomicsAdapter  # noqa: F401
from .pathology_wsi import PathologyWSIAdapter  # noqa: F401
from .synthetic import SyntheticMultiOmicsAdapter  # noqa: F401
from .transcriptomics import TranscriptomicsAdapter  # noqa: F401

__all__ = [
    "FeatureBlock",
    "ModalityAdapter",
    "DicomRadiomicsAdapter",
    "NiftiVolumeAdapter",
    "PathologyWSIAdapter",
    "TranscriptomicsAdapter",
    "GenomicsAdapter",
    "MetabolomicsAdapter",
    "SyntheticMultiOmicsAdapter",
    "MatrixAdapter",
]
