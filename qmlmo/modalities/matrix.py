"""Generic matrix adapter: .npy / .csv / .tsv -> FeatureBlock.

This is the workhorse behind the omics adapters. It exists separately so that
any tabular source can enter the framework without a bespoke class.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter


def _read_delimited(path: str, delimiter: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Read a samples-as-rows table with a header row and an id column."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    if len(rows) < 2:
        raise ValueError(f"{path}: need a header row plus at least one data row")
    header = rows[0]
    feature_names = [h.strip() for h in header[1:]]
    sample_ids: list[str] = []
    data: list[list[float]] = []
    for r in rows[1:]:
        if not r or all(not c.strip() for c in r):
            continue
        if len(r) != len(header):
            raise ValueError(
                f"{path}: row for {r[0]!r} has {len(r)} cells, header has {len(header)}"
            )
        sample_ids.append(r[0].strip())
        data.append([_to_float(c) for c in r[1:]])
    return np.asarray(data, dtype=np.float64), feature_names, sample_ids


def _to_float(cell: str) -> float:
    c = cell.strip()
    if c == "" or c.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return float("nan")
    return float(c)


@register_modality
class MatrixAdapter(ModalityAdapter):
    """Load a pre-built matrix.

    params:
      path            : .npy, .csv or .tsv
      modality        : label written into the FeatureBlock (default "matrix")
      feature_names   : list, or path to a JSON list (required for .npy)
      sample_ids      : list, or path to a JSON list (default s0000...)
      transpose       : set True if the file is features-as-rows
    """

    name = "matrix"

    def load(self) -> FeatureBlock:
        p = self.params
        path = p["path"]
        modality = p.get("modality", "matrix")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{modality}: no such matrix {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            X = np.load(path)
            feature_names = _resolve_list(p.get("feature_names"))
            sample_ids = _resolve_list(p.get("sample_ids"))
        elif ext in {".csv", ".tsv", ".txt"}:
            delim = "\t" if ext in {".tsv", ".txt"} else ","
            X, fn, sid = _read_delimited(path, delim)
            feature_names = _resolve_list(p.get("feature_names")) or fn
            sample_ids = _resolve_list(p.get("sample_ids")) or sid
        else:
            raise ValueError(f"{modality}: unsupported extension {ext!r} for {path}")

        X = np.asarray(X, dtype=np.float64)
        if p.get("transpose"):
            X = X.T

        if feature_names is None:
            raise ValueError(
                f"{modality}: feature_names are required for {ext} input; "
                "pass a list or a path to a JSON list so results stay traceable"
            )
        if sample_ids is None:
            sample_ids = [f"s{i:05d}" for i in range(X.shape[0])]

        return FeatureBlock(
            X=X,
            feature_names=list(feature_names),
            sample_ids=list(sample_ids),
            modality=modality,
            meta={"source": os.path.abspath(path), "adapter": self.name},
        )


def _resolve_list(v: Any) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str) and os.path.exists(v):
        with open(v) as fh:
            return [str(x) for x in json.load(fh)]
    raise ValueError(f"cannot resolve list from {v!r}")
