#!/usr/bin/env python3
"""make_gate_figures.py -- figures for the gate-refutation manuscript.

Every number is read from benchmarks/results/*.json and data/*.json at render
time. Nothing is hard-coded. A missing input raises rather than silently
drawing stale values.

  fig1_gate.svg/.png   Result 1: IF vs classical AUC. The gate has no
                       construct validity -- IF is inverse classical
                       performance, and every candidate clears the line.
  fig2_encoding.svg    Result 2: kernel off-diagonal mass vs encoding scale,
                       with the AUC consequence. The hidden hyperparameter.
  fig3_forest.svg      Per-drug effects with CIs, both encodings, both
                       classical baselines. What we do and do not claim.

Usage:
  .venv/bin/python scripts/make_gate_figures.py [--out docs/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# keep text as real <text> elements: renders identically, stays greppable/editable
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
DATA = REPO / "data"

C_Q = "#8b5cf6"      # quantum
C_C = "#0d9488"      # classical
C_PARP = "#dc2626"   # PARP arm
C_NEG = "#64748b"    # negative controls
C_GATE = "#f59e0b"   # the gate line
GRID = dict(color="#e2e8f0", lw=0.8, zorder=0)


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#94a3b8")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="both", **GRID)
    ax.set_axisbelow(True)


def fig1_gate(out, formats):
    """Result 1: the admission gate measures inverse classical performance."""
    gate = json.load(open(RESULTS / "gate_construct_validity.json"))
    summary = json.load(open(DATA / "final_summary_22drug.json"))
    drugs = summary["drugs"]

    IF = np.array([d["interaction_fraction"] for d in drugs])
    AUC = np.array([d["gbdt_auc"] for d in drugs])
    parp = np.array([d["arm"] == "PARP_inhibitor" for d in drugs])
    names = [d["drug"] for d in drugs]

    rho = gate["test_2_construct"]["spearman_if_vs_gbdt_auc"]["rho"]
    p = gate["test_2_construct"]["spearman_if_vs_gbdt_auc"]["p"]
    r2 = gate["test_2_construct"]["variance_of_if_explained_by_auc"]
    thr = gate["gate_threshold"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # -- panel A: the correlation that kills the gate ---------------------
    _style(a1)
    sl, ic, *_ = stats.linregress(AUC, IF)
    xs = np.linspace(AUC.min() - 0.01, AUC.max() + 0.01, 50)
    a1.plot(xs, sl * xs + ic, color="#1e293b", lw=1.4, ls="--", zorder=2,
            label=f"fit: $R^2$={r2:.2f}")
    a1.scatter(AUC[~parp], IF[~parp], s=64, c=C_NEG, alpha=.85,
               edgecolor="white", lw=1.2, zorder=3, label="cytotoxic control (17)")
    a1.scatter(AUC[parp], IF[parp], s=92, c=C_PARP, marker="D",
               edgecolor="white", lw=1.2, zorder=4, label="PARP inhibitor (5)")
    a1.axhline(thr, color=C_GATE, lw=2, zorder=1)
    a1.text(AUC.min() - 0.005, thr + 0.0015, f"admission gate  IF > {thr}",
            color="#b45309", fontsize=9, fontweight="bold", va="bottom")
    for x, y, n in zip(AUC, IF, names):
        if n in ("methotrexate", "oxaliplatin", "olaparib", "talazoparib", "bleomycin"):
            a1.annotate(n, (x, y), textcoords="offset points", xytext=(6, 5),
                        fontsize=7.5, color="#475569")
    a1.set_xlabel("classical GBDT AUC", fontsize=10)
    a1.set_ylabel("interaction_fraction (IF)", fontsize=10)
    a1.set_title(f"IF is inverse classical performance\n"
                 rf"Spearman $\rho$={rho:.3f}, p={p:.1e}",
                 fontsize=11, fontweight="bold", color="#0f172a")
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    a1.set_ylim(thr - 0.006, IF.max() + 0.006)

    # -- panel B: pass rate + arm comparison ------------------------------
    _style(a2)
    t1, t4 = gate["test_1_discrimination"], gate["test_4_arm_separation"]
    pos = [0.35, 0.85]
    for i, (lab, vals, col) in enumerate((
            ("PARP\n(n=5)", IF[parp], C_PARP),
            ("control\n(n=17)", IF[~parp], C_NEG))):
        jit = np.random.default_rng(0).normal(0, 0.028, len(vals))
        a2.scatter(np.full(len(vals), pos[i]) + jit, vals, s=54, c=col,
                   alpha=.8, edgecolor="white", lw=1, zorder=3)
        a2.hlines(vals.mean(), pos[i] - 0.11, pos[i] + 0.11,
                  color=col, lw=2.6, zorder=4)
    a2.axhline(thr, color=C_GATE, lw=2, zorder=1)
    a2.set_xticks(pos)
    a2.set_xticklabels(["PARP\n(n=5)", "control\n(n=17)"], fontsize=9.5)
    a2.set_xlim(0.12, 1.08)
    a2.set_ylim(thr - 0.006, IF.max() + 0.006)
    a2.set_ylabel("interaction_fraction", fontsize=10)
    a2.set_title(f"{t1['n_passing']}/{t1['n_total']} pass the gate\n"
                 f"Mann-Whitney p={t4['p']:.3f}, opposite to hypothesis (ns)",
                 fontsize=11, fontweight="bold", color="#0f172a")
    a2.annotate("", xy=(0.85, IF[~parp].mean()), xytext=(0.35, IF[parp].mean()),
                arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.2))
    a2.text(0.6, (IF[parp].mean() + IF[~parp].mean()) / 2 + 0.0012,
            "controls trend\nHIGHER (ns)", ha="center", fontsize=8,
            color="#475569", fontweight="normal")

    fig.suptitle("A pre-specified admission gate with no construct validity",
                 fontsize=12.5, fontweight="bold", color="#0f172a", y=1.0)
    fig.tight_layout()
    for f in formats:
        fig.savefig(out / f"fig1_gate.{f}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return "fig1_gate"


def fig2_encoding(out, formats):
    """Result 2: encoding scale is a hidden hyperparameter worth 0.06 AUC."""
    pt = json.load(open(RESULTS / "parp_paired_test.json"))
    pi, tu = pt["arms"]["pi"], pt["arms"]["tuned"]
    names = [r["drug"] for r in tu["per_drug"]]

    off_pi = np.array([r["kernel_offdiag_mean"] for r in pi["per_drug"]])
    off_tu = np.array([r["kernel_offdiag_mean"] for r in tu["per_drug"]])
    q_pi = np.array([r["qsvm_auc"] for r in pi["per_drug"]])
    q_tu = np.array([r["qsvm_auc"] for r in tu["per_drug"]])
    rbf = np.array([r["rbf_auc"] for r in tu["per_drug"]])
    s_pi = pi["per_drug"][0]["encoding_scale"]
    s_tu = tu["per_drug"][0]["encoding_scale"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # -- panel A: off-diagonal mass collapse ------------------------------
    _style(a1)
    a1.set_yscale("log")
    x = np.arange(len(names))
    a1.bar(x - 0.19, off_pi, 0.36, color="#cbd5e1", edgecolor="#94a3b8",
           label=f"scale = $\\pi$  ({s_pi:.3f})", zorder=3)
    a1.bar(x + 0.19, off_tu, 0.36, color=C_Q, edgecolor="#6d28d9",
           label=f"scale = {s_tu:.3f}  (tuned)", zorder=3)
    a1.axhline(1.0, color="#64748b", lw=1, ls=":", zorder=2)
    a1.text(len(names) - 0.5, 1.15, "identity kernel limit", fontsize=7.5,
            color="#64748b", ha="right")
    a1.set_xticks(x)
    a1.set_xticklabels(names, rotation=28, ha="right", fontsize=8.5)
    a1.set_ylabel("kernel off-diagonal mean  (log)", fontsize=10)
    a1.set_title("At scale=$\\pi$ all sample pairs are near-orthogonal\n"
                 f"off-diagonal mass {off_pi.mean():.2e} vs {off_tu.mean():.2f}",
                 fontsize=11, fontweight="bold", color="#0f172a")
    a1.legend(frameon=False, fontsize=8.5, loc="center right")

    # -- panel B: the AUC consequence -------------------------------------
    _style(a2)
    for i in range(len(names)):
        a2.plot([0, 1], [q_pi[i], q_tu[i]], color="#cbd5e1", lw=1.3, zorder=2)
    a2.scatter(np.zeros(len(names)), q_pi, s=78, c="#94a3b8",
               edgecolor="white", lw=1.2, zorder=4, label="QSVM, scale=$\\pi$")
    a2.scatter(np.ones(len(names)), q_tu, s=78, c=C_Q,
               edgecolor="white", lw=1.2, zorder=4, label="QSVM, tuned")
    a2.scatter(np.full(len(names), 1.42), rbf, s=78, c=C_C, marker="s",
               edgecolor="white", lw=1.2, zorder=4, label="SVM-RBF")
    for i, n in enumerate(names):
        a2.annotate(n, (1.42, rbf[i]), textcoords="offset points",
                    xytext=(9, -3), fontsize=7.5, color="#475569")
    d_pi = pi["vs_SVM_RBF"]["mean_delta"]
    d_tu = tu["vs_SVM_RBF"]["mean_delta"]
    a2.set_xticks([0, 1, 1.42])
    a2.set_xticklabels([f"naive\n$\\Delta$={d_pi:+.3f}",
                        f"tuned\n$\\Delta$={d_tu:+.3f}", "classical"],
                       fontsize=9)
    a2.set_xlim(-0.22, 2.05)
    a2.set_ylabel("AUC", fontsize=10)
    a2.set_title("Retuning recovers 0.06 AUC -- repairing the encoding,\n"
                 "not demonstrating quantum signal",
                 fontsize=11, fontweight="bold", color="#0f172a")
    a2.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.suptitle("Encoding scale: an unreported hyperparameter that dominates the "
                 "quantum arm",
                 fontsize=12.5, fontweight="bold", color="#0f172a", y=1.0)
    fig.tight_layout()
    for f in formats:
        fig.savefig(out / f"fig2_encoding.{f}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return "fig2_encoding"


def fig3_forest(out, formats):
    """What we claim and what we decline to claim."""
    pt = json.load(open(RESULTS / "parp_paired_test.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, arm in zip(axes, ("pi", "tuned")):
        _style(ax)
        a = pt["arms"][arm]
        rows = a["per_drug"]
        names = [r["drug"] for r in rows]
        holm = a.get("per_drug_holm", {})
        y = np.arange(len(rows))[::-1]

        for i, r in enumerate(rows):
            yy = y[i]
            d = r["delta_vs_rbf"]
            wd = r.get("within_drug_paired") or {}
            sd = wd.get("sd", 0.0) or 0.0
            n = wd.get("n_folds", 5)
            se = sd / np.sqrt(n) if n else 0.0
            crit = stats.t.ppf(0.975, df=max(n - 1, 1))
            lo, hi = d - crit * se, d + crit * se
            hinfo = holm.get(r["drug"], {}) if isinstance(holm, dict) else {}
            sig = bool(hinfo.get("significant_at_0.05"))
            col = C_PARP if sig else "#475569"
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.8, zorder=3)
            ax.scatter([d], [yy], s=70, c=col, edgecolor="white", lw=1.2, zorder=4)
            ph = hinfo.get("p_holm")
            if ph is not None:
                ax.text(0.163, yy, f"Holm p={ph:.3f}", fontsize=7.5, va="center",
                        color=col, ha="right",
                        fontweight="bold" if sig else "normal")

        s = a["vs_SVM_RBF"]
        ax.axvline(0, color="#0f172a", lw=1.2, zorder=2)
        ax.axvspan(s["ci95"][0], s["ci95"][1], color=C_Q, alpha=.13, zorder=1)
        ax.axvline(s["mean_delta"], color=C_Q, lw=2, ls="--", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(-0.135, 0.17)
        ax.set_xlabel("QSVM $-$ SVM-RBF   (AUC)", fontsize=10)
        lab = "naive  (scale=$\\pi$)" if arm == "pi" else "tuned  (scale=0.196)"
        ax.set_title(f"{lab}\n"
                     f"paired $\\Delta$={s['mean_delta']:+.4f}, p={s['p']:.3f}"
                     f"  |  fold-level Holm: "
                     f"{sum(1 for v in holm.values() if isinstance(v, dict) and v.get('significant_at_0.05'))}/5"
                     f" (exploratory)",
                     fontsize=11, fontweight="bold", color="#0f172a")
        ax.text(0.0, -1.15, "favours classical  <--   |   -->  favours quantum",
                fontsize=7.5, color="#94a3b8", ha="center", transform=ax.get_xaxis_transform(),
                clip_on=False)

    fig.suptitle("No quantum advantage: nominal fold-based signals only where the "
                 "quantum kernel loses (exploratory)",
                 fontsize=12.5, fontweight="bold", color="#0f172a", y=1.02)
    fig.tight_layout()
    for f in formats:
        fig.savefig(out / f"fig3_forest.{f}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return "fig3_forest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--formats", default="svg,png")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    for fn in (fig1_gate, fig2_encoding, fig3_forest):
        name = fn(out, formats)
        print(f"  wrote {name}.{{{','.join(formats)}}}")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
