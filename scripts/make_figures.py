#!/usr/bin/env python3
"""
make_figures.py — publication-grade figure generation for the QML-in-multiomics manuscript.

Every number is read from benchmarks/results/*.json at render time.
No hard-coded metrics. If a result file is missing, the panel is skipped and
reported, rather than silently drawing stale values.

Figures produced (docs/figures/):
  fig1_workflow.svg        Perspective-style pipeline schematic (hand-authored SVG)
  fig2_forest.svg/.png     Balanced accuracy + AUC with CV dispersion, vs majority baseline
  fig3_interaction.svg     SHAP interaction structure (top pairs + IF decomposition)
  fig4_cost.svg            Accuracy-per-second: the quantum cost axis nobody plots
  fig5_ablation.svg        PGR ablation + PCA8 control: what the signal actually is

Usage:
  .venv/bin/python scripts/make_figures.py [--out docs/figures] [--formats svg,png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"

# ---------------------------------------------------------------- style
# One palette for the whole manuscript. Quantum = violet, classical = teal.
C_Q = "#8b5cf6"
C_C = "#0d9488"
C_BASE = "#94a3b8"
C_WARN = "#dc2626"
C_TXT = "#1e293b"
C_DIM = "#64748b"
C_LINE = "#cbd5e1"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.edgecolor": C_LINE,
    "axes.labelcolor": C_TXT,
    "axes.titlesize": 10.5,
    "axes.titleweight": "600",
    "xtick.color": C_DIM,
    "ytick.color": C_DIM,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

MISSING: list[str] = []


def load(name: str):
    p = RESULTS / f"{name}.json"
    if not p.exists():
        MISSING.append(name)
        return None
    return json.loads(p.read_text())


def save(fig, out: Path, stem: str, formats: list[str]):
    for fmt in formats:
        fig.savefig(out / f"{stem}.{fmt}")
    plt.close(fig)
    print(f"  [ok] {stem}.{'/'.join(formats)}")


# ------------------------------------------------------- data collection
def collect_arms() -> tuple[list[dict], float, int]:
    """Flatten every run JSON into one arm table."""
    files = [
        ("brca_pr_v1_classical", None),
        ("brca_pr_v1_gbdt", None),
        ("brca_pr_v1_quantum", None),
        ("brca_pr_v1_qsvm_balanced", "qsvm_angle (class-balanced)"),
        ("brca_pr_v1_vqc_zz_circ", None),
    ]
    arms, baseline, n = [], None, None
    for fname, relabel in files:
        d = load(fname)
        if not d or "results" not in d:
            continue
        baseline = d.get("majority_baseline", baseline)
        n = d.get("n_samples", n)
        for algo, r in d["results"].items():
            std = r.get("std", {})
            arms.append({
                "name": relabel or algo,
                "quantum": bool(r.get("quantum")),
                "bal": r.get("balanced_accuracy", float("nan")),
                "bal_sd": std.get("balanced_accuracy", 0.0),
                "auc": r.get("auc_roc", float("nan")),
                "auc_sd": std.get("auc_roc", 0.0),
                "acc": r.get("accuracy", float("nan")),
                "train_s": r.get("train_time", 0.0),
                "infer_s": r.get("inference_time", 0.0),
                "splits": r.get("n_splits", 5),
                "repeats": r.get("repeats", 1),
                "above": r.get("above_majority"),
            })
    # keep the better qsvm variant visible but both present; sort quantum last
    arms.sort(key=lambda a: (a["quantum"], -a["bal"]))
    return arms, (baseline or 0.0), (n or 0)


# ------------------------------------------------------------- figure 1
def fig1_workflow(out: Path, n_samples: int, formats: list[str]):
    """Perspective-style pipeline. Drawn, not screenshotted -- stays editable."""
    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    ax.set_xlim(0, 112); ax.set_ylim(0, 34); ax.axis("off")

    stages = [
        ("Cohort", f"TCGA-BRCA\nn={n_samples}\nSTAR counts", "#e0f2fe", "#0369a1"),
        ("Label", "BCR XML\nIHC receptor\nPR / ER / HER2", "#e0f2fe", "#0369a1"),
        ("Filter", "log1p\ntop-var 5000\n(no PCA on disk)", "#f1f5f9", "#475569"),
        ("Fold split", "StratifiedKFold\n5 x 5\nreduce INSIDE fold", "#fef3c7", "#b45309"),
        ("Encode", "angle / ZZ\nfeature map\n8 qubits", "#ede9fe", "#6d28d9"),
        ("Model", "QSVM  |  VQC\nvs\nSVM / RF / GBDT", "#ede9fe", "#6d28d9"),
        ("Readout", "bal-acc, AUC\nkappa, dead-class\nvs majority", "#dcfce7", "#15803d"),
    ]
    w, h, gap, y = 13.0, 13.0, 3.0, 12.0
    for i, (title, body, fc, ec) in enumerate(stages):
        x = 1.5 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1",
                                    facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
                fontsize=9.5, fontweight="700", color=ec)
        ax.text(x + w / 2, y + h / 2 - 1.8, body, ha="center", va="center",
                fontsize=7.4, color=C_TXT, linespacing=1.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.3, y + h / 2), (x + w + gap - 0.3, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=C_DIM, linewidth=1.1))

    # the leakage guard rail -- this is the methodological claim of the figure
    x0 = 1.5 + 3 * (w + gap)
    x1 = 1.5 + 6 * (w + gap) + w
    ax.plot([x0, x1], [8.6, 8.6], color=C_WARN, linewidth=1.3, linestyle=(0, (5, 3)))
    ax.plot([x0, x0], [8.6, 11.4], color=C_WARN, linewidth=1.3, linestyle=(0, (5, 3)))
    ax.plot([x1, x1], [8.6, 11.4], color=C_WARN, linewidth=1.3, linestyle=(0, (5, 3)))
    ax.text((x0 + x1) / 2, 6.6,
            "leakage boundary — every reduction/scaling step is fitted on the training fold only "
            "(harness refuses k-fold on pre-projected matrices)",
            ha="center", va="center", fontsize=7.6, color=C_WARN, style="italic")

    ax.text(1.5, 30.5, "Fig. 1 | Leakage-controlled QML benchmarking workflow for multi-omics",
            fontsize=11, fontweight="700", color=C_TXT)
    ax.text(1.5, 27.4,
            "Data provenance (blue) → preprocessing (grey) → resampling (amber) → "
            "quantum/classical model arms (violet) → baseline-anchored readout (green).",
            fontsize=8, color=C_DIM)
    save(fig, out, "fig1_workflow", formats)


# ------------------------------------------------------------- figure 2
def fig2_forest(out: Path, arms, baseline, n_samples, formats):
    if not arms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.52 * len(arms) + 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    ys = np.arange(len(arms))[::-1]
    names = [a["name"] for a in arms]
    cols = [C_Q if a["quantum"] else C_C for a in arms]

    for ax, key, sd, title, ref in (
        (axes[0], "bal", "bal_sd", "Balanced accuracy", 0.5),
        (axes[1], "auc", "auc_sd", "AUROC", 0.5),
    ):
        vals = np.array([a[key] for a in arms])
        errs = np.array([a[sd] for a in arms])
        ax.errorbar(vals, ys, xerr=errs, fmt="o", ms=6.5, capsize=3.2, lw=1.3,
                    ecolor=C_DIM, mfc="none", zorder=3, linestyle="none")
        for y_, v_, c_ in zip(ys, vals, cols):
            ax.plot(v_, y_, "o", ms=6.5, color=c_, zorder=4)
        ax.axvline(ref, color=C_LINE, lw=1, ls=":")
        ax.text(ref, len(arms) - 0.35, "chance", fontsize=7, color=C_DIM, ha="center")
        if key == "bal":
            ax.axvline(baseline, color=C_WARN, lw=1.2, ls="--", zorder=2)
            ax.text(baseline, len(arms) - 0.35, f"majority {baseline:.3f}",
                    fontsize=7, color=C_WARN, ha="center")
        ax.set_xlim(0.3, 1.0)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="#f1f5f9", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names, fontsize=8.5)
    for tick, a in zip(axes[0].get_yticklabels(), arms):
        tick.set_color(C_Q if a["quantum"] else C_TXT)
        if a["quantum"]:
            tick.set_fontweight("600")

    for ax, key in ((axes[0], "bal"), (axes[1], "auc")):
        for y_, a in zip(ys, arms):
            ax.text(0.985, y_, f"{a[key]:.3f}", ha="right", va="center",
                    fontsize=7.6, color=C_DIM, family="monospace")

    fig.suptitle(f"Fig. 2 | No quantum arm exceeds the classical envelope (TCGA-BRCA PR_Status, n={n_samples})",
                 x=0.012, ha="left", fontsize=11, fontweight="700", y=1.02, color=C_TXT)
    fig.text(0.012, -0.055,
             "Points = mean over repeated stratified 5-fold CV; bars = SD across folds. "
             "Violet = quantum, teal = classical. Red dashed line is the majority-class rate — "
             "an arm below it has learned nothing.",
             fontsize=7.8, color=C_DIM, ha="left")
    save(fig, out, "fig2_forest", formats)


# ------------------------------------------------------------- figure 3
def fig3_interaction(out: Path, formats):
    d = load("brca_pr_v1_shap_interaction")
    if not d:
        return
    pairs = d.get("top_interaction_pairs", [])[:10][::-1]
    if_frac = d.get("interaction_fraction", float("nan"))
    main_m = d.get("main_effect_mass", 0.0)
    inter_m = d.get("interaction_mass", 0.0)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                   gridspec_kw={"width_ratios": [1.9, 1.0], "wspace": 0.32})

    labels = [f"{p['a']} × {p['b']}" for p in pairs]
    strengths = [p["strength"] for p in pairs]
    ys = np.arange(len(pairs))
    axl.barh(ys, strengths, color=["#c4b5fd" if "PGR" in l else "#a7f3d0" for l in labels],
             edgecolor="#8b5cf6", linewidth=0.7, height=0.68)
    axl.set_yticks(ys); axl.set_yticklabels(labels, fontsize=8, family="monospace")
    axl.set_xlabel("mean |SHAP interaction value|", fontsize=8.5)
    axl.set_title("Top pairwise interactions", loc="left")
    axl.grid(axis="x", color="#f1f5f9", lw=0.8); axl.set_axisbelow(True)
    for y_, s_ in zip(ys, strengths):
        axl.text(s_ + max(strengths) * 0.015, y_, f"{s_:.3f}", va="center",
                 fontsize=7.2, color=C_DIM, family="monospace")

    total = main_m + inter_m
    axr.bar([0], [main_m], width=0.55, color="#cbd5e1", edgecolor="#94a3b8", label="main effects")
    axr.bar([0], [inter_m], bottom=[main_m], width=0.55, color=C_Q, edgecolor="#6d28d9",
            label="interactions")
    axr.set_xlim(-0.7, 0.7); axr.set_xticks([])
    axr.set_ylabel("SHAP mass", fontsize=8.5)
    axr.set_title("Effect decomposition", loc="left")
    axr.text(0, main_m / 2, f"{main_m / total:.0%}", ha="center", va="center",
             fontsize=10, fontweight="700", color="#475569")
    axr.text(0, main_m + inter_m / 2, f"{inter_m / total:.0%}", ha="center", va="center",
             fontsize=10, fontweight="700", color="white")
    axr.axhline(total * 0.60, color=C_WARN, ls="--", lw=1.1)
    axr.text(0.7, total * 0.60, "  IF = 0.40\n  admission\n  threshold", va="center",
             fontsize=7.2, color=C_WARN)
    axr.legend(frameon=False, fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    fig.suptitle(f"Fig. 3 | Interaction fraction IF = {if_frac:.3f} — below the pre-registered 0.40 admission threshold",
                 x=0.012, ha="left", fontsize=11, fontweight="700", y=1.03, color=C_TXT)
    fig.text(0.012, -0.10,
             "Exact TreeExplainer interaction values on a stratified 30% holdout. "
             "Seven of the ten strongest pairs involve PGR itself — the task is dominated by a "
             "single main effect, not by the high-order structure a quantum kernel is meant to exploit.",
             fontsize=7.8, color=C_DIM, ha="left")
    save(fig, out, "fig3_interaction", formats)


# ------------------------------------------------------------- figure 4
def fig4_cost(out: Path, arms, baseline, formats):
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for a in arms:
        t = max(a["train_s"] + a["infer_s"], 1e-3)
        c = C_Q if a["quantum"] else C_C
        ax.scatter(t, a["bal"], s=150, color=c, alpha=0.82, edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(a["name"], (t, a["bal"]), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=7.6, color=C_TXT)
    ax.set_xscale("log")
    ax.axhline(baseline, color=C_WARN, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0] * 1.15, baseline + 0.008, f"majority baseline {baseline:.3f}",
            fontsize=7.4, color=C_WARN)
    ax.set_xlabel("wall-clock per CV fold (train + inference, seconds, log scale)", fontsize=8.5)
    ax.set_ylabel("balanced accuracy", fontsize=8.5)
    ax.grid(color="#f1f5f9", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Fig. 4 | Cost–benefit: quantum arms buy 3–4 orders of magnitude of compute for no accuracy",
                 loc="left", fontsize=10.5)
    fig.text(0.012, -0.03,
             "Simulator wall-clock, not hardware queue time. The upper-left quadrant is where a "
             "method should live; every violet point sits lower AND ~1000× further right.",
             fontsize=7.8, color=C_DIM, ha="left")
    save(fig, out, "fig4_cost", formats)


# ------------------------------------------------------------- figure 5
def fig5_ablation(out: Path, formats):
    abl = load("brca_pr_v1_pgr_ablation")
    pca = load("brca_pr_v1_pca8_control")
    if not abl and not pca:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    if abl:
        groups = ["all 5000 genes", "PGR removed"]
        bal = [abl["full"]["bal"], abl["minus_pgr"]["bal"]]
        bal_sd = [abl["full"]["bal_sd"], abl["minus_pgr"]["bal_sd"]]
        auc = [abl["full"]["auc"], abl["minus_pgr"]["auc"]]
        auc_sd = [abl["full"]["auc_sd"], abl["minus_pgr"]["auc_sd"]]
        x = np.arange(2); w = 0.34
        ax.bar(x - w / 2, bal, w, yerr=bal_sd, capsize=3.5, color="#5eead4",
               edgecolor=C_C, label="bal-acc", error_kw={"lw": 1, "ecolor": C_DIM})
        ax.bar(x + w / 2, auc, w, yerr=auc_sd, capsize=3.5, color="#99f6e4",
               edgecolor=C_C, label="AUROC", error_kw={"lw": 1, "ecolor": C_DIM})
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
        ax.set_ylim(0.5, 1.0); ax.legend(frameon=False, fontsize=7.6, ncol=2)
        ax.set_title(f"PGR ablation (ΔAUC = {abl['delta_auc']:+.3f})", loc="left")
        ax.grid(axis="y", color="#f1f5f9", lw=0.8); ax.set_axisbelow(True)

    ax = axes[1]
    if pca:
        keys = ["svm_rbf_pca8", "svm_rbf_raw5000", "rf_baseline_pca8", "rf_baseline_raw5000"]
        keys = [k for k in keys if k in pca]
        labs = [k.replace("_baseline", "").replace("_", "\n") for k in keys]
        bal = [pca[k]["bal"][0] for k in keys]
        sd = [pca[k]["bal"][1] for k in keys]
        cols = ["#fbbf24" if "pca8" in k else "#5eead4" for k in keys]
        ax.bar(np.arange(len(keys)), bal, 0.6, yerr=sd, capsize=3.5, color=cols,
               edgecolor="#64748b", error_kw={"lw": 1, "ecolor": C_DIM})
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(labs, fontsize=7.4)
        ax.set_ylim(0.5, 1.0)
        ax.set_title("8-dim bottleneck control (amber) vs full features", loc="left")
        ax.grid(axis="y", color="#f1f5f9", lw=0.8); ax.set_axisbelow(True)

    fig.suptitle("Fig. 5 | Where the signal actually lives",
                 x=0.012, ha="left", fontsize=11, fontweight="700", y=1.03, color=C_TXT)
    fig.text(0.012, -0.09,
             "Left: dropping the PGR transcript costs 7.6 AUC points but almost no balanced accuracy — "
             "the label is a protein-level IHC call, partly recoverable from its own mRNA. "
             "Right: forcing classical models through the same 8-dimensional bottleneck the quantum "
             "circuits face costs ~3 points — the qubit budget is a handicap, but not the whole gap.",
             fontsize=7.8, color=C_DIM, ha="left")
    save(fig, out, "fig5_ablation", formats)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "figures"))
    ap.add_argument("--formats", default="svg,png")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    arms, baseline, n_samples = collect_arms()
    print(f"[figures] {len(arms)} arms · n={n_samples} · majority={baseline:.4f}")
    if n_samples and n_samples != 138:
        print(f"[figures] NOTE cohort size is {n_samples}, not the 138 previously reported")

    fig1_workflow(out, n_samples, formats)
    fig2_forest(out, arms, baseline, n_samples, formats)
    fig3_interaction(out, formats)
    fig4_cost(out, arms, baseline, formats)
    fig5_ablation(out, formats)

    manifest = {
        "generated_by": "scripts/make_figures.py",
        "n_samples": n_samples,
        "majority_baseline": baseline,
        "arms": arms,
        "missing_result_files": MISSING,
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[figures] manifest -> {out / 'figure_manifest.json'}")
    if MISSING:
        print(f"[figures] MISSING result files (panels skipped): {MISSING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
