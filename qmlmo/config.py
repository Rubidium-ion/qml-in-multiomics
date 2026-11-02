"""YAML experiment spec -> loaded blocks + labels + groups + RunConfig.

A config is the reproducibility unit. Everything the runner needs is declared
here, so a result JSON can always be traced back to one file.

Labels are resolved AFTER modality alignment, not before. This matters for real
cohorts: pooled CPTAC drops cases that lack one modality, so a label vector
built against the pre-alignment sample order would be silently misaligned by a
few rows -- the worst kind of bug, because everything still runs.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from . import modalities as _modalities  # noqa: F401  (registers adapters)
from .integrate import build_dataset
from .modalities.base import FeatureBlock
from .modalities.synthetic import SyntheticMultiOmicsAdapter
from .registry import build_modality
from .runner import RunConfig


def load_yaml(path: str) -> dict[str, Any]:
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def load_blocks(cfg: dict[str, Any]) -> tuple[list[FeatureBlock], np.ndarray | None, dict]:
    """Build every declared modality block.

    Synthetic experiments are special-cased: the generator produces blocks and
    labels together, since the labels are defined by the same generative process.
    """
    extra: dict[str, Any] = {}

    if "synthetic" in cfg:
        gen = SyntheticMultiOmicsAdapter(**cfg["synthetic"])
        blocks, y, truth = gen.generate()
        extra["synthetic_ground_truth"] = truth
        return blocks, y, extra

    mods = cfg.get("modalities")
    if not mods:
        raise ValueError("config needs either `synthetic:` or `modalities:`")

    blocks = []
    for entry in mods:
        entry = dict(entry)
        kind = entry.pop("type", None) or entry.pop("adapter", None)
        if not kind:
            raise ValueError(f"modality entry missing `type`: {entry}")
        blocks.append(build_modality(kind, **entry).load())
    return blocks, None, extra


def _read_label_table(cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, str] | None]:
    """Return (id -> raw label, id -> group) from the labels section.

    Group extraction happens here rather than in a second file read, because a
    group column that silently disappears is exactly how a leave-one-group-out
    run degrades into a random split without anyone noticing.
    """
    lab = cfg.get("labels")
    if not lab:
        raise ValueError("config needs a `labels:` section (path + optional binarize)")

    path = lab.get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"labels: {path!r} not found")

    group_col = lab.get("group_column") or (cfg.get("run", {}) or {}).get("group_column")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path) as fh:
            obj = json.load(fh)
        if not isinstance(obj, dict):
            raise ValueError(
                f"labels: {path} is a bare JSON list, which has no sample ids. "
                "Use a mapping {sample_id: label} or a CSV."
            )
        if group_col:
            raise ValueError(
                "labels.group_column needs a CSV/TSV table; JSON mapping carries "
                "labels only"
            )
        return {str(k): _num(v) for k, v in obj.items()}, None

    if ext in {".csv", ".tsv"}:
        import csv
        delim = "\t" if ext == ".tsv" else ","
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=delim))
        if not rows:
            raise ValueError(f"labels: {path} has no data rows")

        cols = list(rows[0])
        id_col = lab.get("id_column") or cols[0]
        y_col = lab.get("label_column") or cols[-1]
        for c, what in ((id_col, "id_column"), (y_col, "label_column")):
            if c not in cols:
                raise ValueError(f"labels: {what}={c!r} not in {path}; columns are {cols}")

        groups = None
        if group_col:
            if group_col not in cols:
                raise ValueError(
                    f"labels.group_column={group_col!r} not in {path}; columns are {cols}. "
                    "Group-aware CV cannot fall back to a random split silently."
                )
            groups = {str(r[id_col]): str(r[group_col]) for r in rows}

        return {str(r[id_col]): _num(r[y_col]) for r in rows}, groups

    if ext == ".npy":
        raise ValueError(
            "labels: .npy has no sample ids, so it cannot be aligned after modality "
            "intersection. Provide a CSV/JSON keyed by sample id."
        )
    raise ValueError(f"labels: unsupported extension {ext}")


def _num(v: Any) -> float:
    s = str(v).strip()
    if s in {"", "NA", "NaN", "nan", "None", "NULL"}:
        return float("nan")
    return float(s)


def _binarize(y: np.ndarray, spec: Any) -> np.ndarray:
    """`quantile:25` or `threshold:0.5`. Records nothing silently."""
    s = str(spec)
    if s.startswith("quantile:"):
        q = float(s.split(":", 1)[1]) / 100.0
        return (y <= float(np.nanquantile(y, q))).astype(np.int32)
    if s.startswith("threshold:"):
        t = float(s.split(":", 1)[1])
        return (y > t).astype(np.int32)
    raise ValueError(
        f"labels.binarize: expected 'quantile:N' or 'threshold:X', got {spec!r}")


def build_run_config(cfg: dict[str, Any]) -> RunConfig:
    r = dict(cfg.get("run", {}))
    models = r.pop("models", None) or {"svm_rbf": {}, "gbdt": {}, "qsvm_fidelity": {}}
    if isinstance(models, list):
        models = {m: {} for m in models}

    prot = r.pop("protected_features", []) or []
    if isinstance(prot, str) and os.path.exists(prot):
        with open(prot) as fh:
            prot = [str(x) for x in json.load(fh)]

    cv = str(r.pop("cv", "stratified"))
    group_column = r.pop("group_column", None) or \
        (cfg.get("labels", {}) or {}).get("group_column")

    if cv in {"grouped", "leave_one_group_out"} and not group_column:
        raise ValueError(
            f"run.cv={cv!r} requires `group_column` naming a column in the labels "
            "table (e.g. cohort, centre, subject_id)."
        )

    return RunConfig(
        models=models,
        n_splits=int(r.pop("n_splits", 5)),
        seed=int(r.pop("seed", 42)),
        normalize=r.pop("normalize", "zscore"),
        encoding_scale=_scale(r.pop("encoding_scale", None)),
        select_k=r.pop("select_k", None),
        selector=r.pop("selector", "gbdt"),
        protected_features=list(prot),
        modality_quotas={str(k): int(v) for k, v in
                         (r.pop("modality_quotas", {}) or {}).items()},
        cv=cv,
        group_column=group_column,
    )


def _scale(v: Any) -> float | None:
    """Accepts a number, or 'pi', 'pi/16' — because the default matters here."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(" ", "")
    if s == "pi":
        return float(np.pi)
    if s.startswith("pi/"):
        return float(np.pi) / float(s[3:])
    return float(s)


def prepare(cfg: dict[str, Any]):
    """config dict -> (FusedDataset, y, RunConfig, extra)

    `extra["groups"]` holds the aligned group vector when a group column is
    configured, else None.
    """
    blocks, y_syn, extra = load_blocks(cfg)
    max_missing = float(cfg.get("integration", {}).get("max_missing", 0.2))
    fused, report = build_dataset(blocks, max_missing=max_missing)
    run_cfg = build_run_config(cfg)

    groups = None
    if y_syn is not None:
        # synthetic: labels are positional and no rows are dropped
        if len(y_syn) != fused.n_samples:
            raise ValueError(
                f"synthetic generator gave {len(y_syn)} labels for "
                f"{fused.n_samples} aligned samples")
        y = np.asarray(y_syn).ravel()
    else:
        lab_map, group_map = _read_label_table(cfg)
        ids = fused.sample_ids
        missing = [s for s in ids if s not in lab_map]
        if missing:
            keep_idx = [i for i, s in enumerate(ids) if s in lab_map]
            if not keep_idx:
                raise KeyError(
                    f"labels: none of {len(ids)} aligned samples have a label, "
                    f"e.g. {missing[:3]}"
                )
            from .integrate import FusedDataset
            fused = FusedDataset(
                X=fused.X[np.asarray(keep_idx)],
                feature_names=list(fused.feature_names),
                sample_ids=[ids[i] for i in keep_idx],
                block_slices=dict(fused.block_slices),
                meta={**fused.meta, "dropped_unlabelled": missing},
            )
            extra["dropped_unlabelled"] = missing
            ids = fused.sample_ids
        y = np.asarray([lab_map[s] for s in ids], dtype=np.float64)

        binz = (cfg.get("labels") or {}).get("binarize")
        if binz:
            y = _binarize(y, binz)
        elif np.isnan(y).any():
            n_nan = int(np.isnan(y).sum())
            raise ValueError(
                f"labels: {n_nan} sample(s) have a missing label and no `binarize` "
                "rule to interpret them. Drop them upstream or fix the table.")

        if run_cfg.group_column:
            if not group_map:
                raise ValueError(
                    f"run.cv={run_cfg.cv!r} wants group_column="
                    f"{run_cfg.group_column!r} but the labels table produced no groups")
            gmissing = [s for s in ids if s not in group_map]
            if gmissing:
                raise KeyError(
                    f"labels: {len(gmissing)} sample(s) lack a "
                    f"{run_cfg.group_column!r} value, e.g. {gmissing[:3]}")
            groups = [group_map[s] for s in ids]

    extra["integration_report"] = report
    extra["config_path"] = cfg.get("_config_path")
    extra["groups"] = groups
    return fused, np.asarray(y).ravel(), run_cfg, extra
