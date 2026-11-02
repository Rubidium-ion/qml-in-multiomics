"""Command line entry point.

    python -m qmlmo run     configs/exp_synthetic_crossmodal.yaml -o results/x.json
    python -m qmlmo sweep   configs/exp_synthetic_crossmodal.yaml --scales pi pi/4 pi/16
    python -m qmlmo inspect configs/exp_synthetic_crossmodal.yaml
    python -m qmlmo list
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _cmd_list(_args) -> int:
    from . import modalities as _m  # noqa: F401
    from .models import list_models
    from .registry import list_modalities

    print("modalities:", ", ".join(list_modalities()))
    print("models:    ", ", ".join(list_models()))
    print("cv:         stratified, grouped, leave_one_group_out")
    return 0


def _cmd_inspect(args) -> int:
    from .config import load_yaml, prepare
    from .splits import audit_group_label_balance

    fused, y, cfg, extra = prepare(load_yaml(args.config))
    counts = np.bincount(np.asarray(y, dtype=int))
    out = {
        "dataset": fused.summary(),
        "labels": {"n": int(y.size), "counts": counts.tolist(),
                   "majority_baseline": float(counts.max() / counts.sum())},
        "models": list(cfg.models),
        "cv": cfg.cv,
        "group_column": cfg.group_column,
        "encoding_scale": cfg.encoding_scale,
        "select_k": cfg.select_k,
        "modality_quotas": cfg.modality_quotas,
        "integration": extra.get("integration_report"),
    }
    if extra.get("groups") is not None:
        out["group_audit"] = audit_group_label_balance(y, extra["groups"])
    print(json.dumps(out, indent=2, default=str))
    return 0


def _print_results(payload: dict, cfg) -> None:
    n_folds = payload.get("cv_plan", {}).get("n_folds", cfg.n_splits)
    for name, r in payload["results"].items():
        auc = r.get("auc")
        line = f"  {name:16s} " + (f"auc={auc:.4f}" if auc is not None else "auc=n/a  ")
        line += f"  bal_acc={r['balanced_accuracy']:.4f}"
        if r.get("kernel_offdiag_mean") is not None:
            line += f"  offdiag={r['kernel_offdiag_mean']:.4g}"
            if r.get("degenerate_folds"):
                line += f"  DEGENERATE({r['degenerate_folds']}/{n_folds})"
        if r.get("folds_without_auc"):
            line += f"  [no AUC: {','.join(map(str, r['folds_without_auc']))}]"
        print(line)

    print(f"  majority baseline = {payload['labels']['majority_baseline']:.4f}")

    plan = payload.get("cv_plan", {})
    if plan.get("kind") and plan["kind"] != "stratified":
        print(f"  cv = {plan['kind']} over {plan['n_folds']} group(s): "
              f"{', '.join(map(str, plan.get('fold_labels', [])[:8]))}")
        for name, r in payload["results"].items():
            pg = r.get("per_group")
            if not pg:
                continue
            parts = [f"{g}={v['auc']:.3f}" if v.get("auc") is not None else f"{g}=n/a"
                     for g, v in pg.items()]
            print(f"    {name:14s} per-group auc: {'  '.join(parts)}")

    ga = payload.get("group_audit")
    if ga:
        print(f"  group prevalence spread = {ga.get('prevalence_spread')}")
        if ga.get("warning"):
            print(f"  ! {ga['warning']}")
    if payload.get("interpretation_warning"):
        print(f"  ! {payload['interpretation_warning']}")


def _cmd_run(args) -> int:
    from .config import load_yaml, prepare
    from .runner import run_cv, save_results

    raw = load_yaml(args.config)
    fused, y, cfg, extra = prepare(raw)

    if args.encoding_scale is not None:
        from .config import _scale
        cfg.encoding_scale = _scale(args.encoding_scale)
    if args.cv:
        cfg.cv = args.cv
        if args.cv in {"grouped", "leave_one_group_out"} and extra.get("groups") is None:
            raise SystemExit(
                f"--cv {args.cv} needs group labels; add `group_column` to the "
                "labels section of the config."
            )

    payload = run_cv(fused, y, cfg, groups=extra.get("groups"))
    payload["extra"] = {k: v for k, v in extra.items() if k != "groups"}

    out = args.out or os.path.join(
        "benchmarks", "results",
        os.path.splitext(os.path.basename(args.config))[0] + ".json")
    path = save_results(payload, out)

    print(f"[qmlmo] {fused.n_samples} samples x {fused.n_features} features -> {path}")
    _print_results(payload, cfg)
    return 0


def _cmd_sweep(args) -> int:
    """Encoding-scale sweep: the diagnostic the manuscript argues is mandatory."""
    from .config import _scale, load_yaml, prepare
    from .runner import run_cv, save_results

    raw = load_yaml(args.config)
    fused, y, cfg, extra = prepare(raw)
    if args.cv:
        cfg.cv = args.cv
    groups = extra.get("groups")

    rows = []
    for s in args.scales:
        cfg.encoding_scale = _scale(s)
        payload = run_cv(fused, y, cfg, groups=groups)
        for name, r in payload["results"].items():
            if not r.get("quantum"):
                continue
            rows.append({
                "scale_label": s,
                "encoding_scale": cfg.encoding_scale,
                "model": name,
                "auc": r.get("auc"),
                "auc_sd": r.get("auc_sd"),
                "kernel_offdiag_mean": r.get("kernel_offdiag_mean"),
                "degenerate_folds": r.get("degenerate_folds"),
                "per_group": r.get("per_group"),
            })
            auc = r.get("auc")
            print(f"[sweep] scale={s:8s} {name}: "
                  + (f"auc={auc:.4f}" if auc is not None else "auc=n/a")
                  + f"  offdiag={r.get('kernel_offdiag_mean'):.4g}"
                  + ("  DEGENERATE" if r.get("degenerate_folds") else ""))

    best = max((r for r in rows if r["auc"] is not None),
               key=lambda r: r["auc"], default=None)
    if best:
        print(f"\n[sweep] best: scale={best['scale_label']} auc={best['auc']:.4f} "
              f"offdiag={best['kernel_offdiag_mean']:.4g}")
        print("  Report this scale with the result. It is a property of THIS dataset "
              "and must not be inherited from another paper.")

    out = args.out or os.path.join(
        "benchmarks", "results",
        os.path.splitext(os.path.basename(args.config))[0] + "_scale_sweep.json")
    path = save_results({"sweep": rows, "best": best, "cv": cfg.cv,
                         "dataset": fused.summary()}, out)
    print(f"[qmlmo] sweep -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qmlmo", description="multimodal QML experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cv_choices = ["stratified", "grouped", "leave_one_group_out"]

    p = sub.add_parser("run", help="run one cross-validated experiment")
    p.add_argument("config")
    p.add_argument("-o", "--out")
    p.add_argument("--encoding-scale", dest="encoding_scale")
    p.add_argument("--cv", choices=cv_choices, help="override run.cv")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("sweep", help="sweep encoding scale, report kernel degeneracy")
    p.add_argument("config")
    p.add_argument("--scales", nargs="+",
                   default=["pi", "pi/2", "pi/4", "pi/8", "pi/16"])
    p.add_argument("--cv", choices=cv_choices)
    p.add_argument("-o", "--out")
    p.set_defaults(fn=_cmd_sweep)

    p = sub.add_parser("inspect", help="load config, print shapes, run nothing")
    p.add_argument("config")
    p.set_defaults(fn=_cmd_inspect)

    p = sub.add_parser("list", help="list registered modalities, models, cv modes")
    p.set_defaults(fn=_cmd_list)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
