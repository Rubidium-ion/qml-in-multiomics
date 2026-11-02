#!/usr/bin/env python3
"""Assemble the gate-refutation manuscript from its four verified source pieces.

Sources (all drafted 2026-08-09, all numeric claims programmatically verified):
  docs/abstract.md                     -> abstract (long + short)
  docs/intro_discussion_gate_refutation.md -> Introduction, Discussion, References
  docs/methods_gate_refutation.md      -> Methods
  docs/paper_draft_gate_refutation.md  -> Results (Result 1, Result 2, What we do not claim)

Output: docs/manuscript/manuscript_gate_refutation.md
Cross-references are renumbered for the assembled layout (§3.x results, §4.x discussion).
Figures F1-F3 are embedded with unified captions. Rerunnable: always regenerates from sources.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

ABSTRACT = DOCS / "abstract.md"
INTRO = DOCS / "intro_discussion_gate_refutation.md"
METHODS = DOCS / "methods_gate_refutation.md"
RESULTS = DOCS / "paper_draft_gate_refutation.md"
OUT = DOCS / "manuscript" / "manuscript_gate_refutation.md"

TITLE = "A pre-registered interaction-fraction criterion fails to identify tasks suited to quantum kernels"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end].rstrip()


def strip_first_heading(text: str, heading: str) -> str:
    """Drop the source section heading itself, keeping the body."""
    idx = text.index(heading)
    rest = text[idx + len(heading):]
    return rest.lstrip("\n")


def fix_crossrefs(text: str) -> str:
    # Results/discussion sections are renumbered in the assembled layout.
    text = text.replace("§Result 1", "§3.1")
    text = text.replace("§Result 2", "§3.2")
    text = text.replace("§What we do not claim", "§3.3")
    text = text.replace('§"What we do not claim"', "§3.3")
    text = text.replace("(§5 of the manuscript)", "(§4.2)")
    return text


def main() -> int:
    abstract = read(ABSTRACT)
    intro_disc = read(INTRO)
    methods = read(METHODS)
    results = read(RESULTS)

    # ---------- abstract ----------
    # structured (~250 words) + short (~120 words), keep notes out of the manuscript
    abst_body = slice_between(abstract, "## Structured abstract (~250 words)", "## Short abstract")
    abst_body = abst_body.replace("## Structured abstract (~250 words)", "").strip()
    short_body = slice_between(abstract, "## Short abstract (~120 words", "## Notes for assembly")
    short_body = strip_first_heading(short_body, "## Short abstract (~120 words, for abstract-only submissions)").strip()
    short_body = short_body.rstrip("-\n ").rstrip()

    # ---------- introduction ----------
    intro_body = slice_between(intro_disc, "# 1. Introduction", "# 2. Discussion")
    intro_body = fix_crossrefs(strip_first_heading(intro_body, "# 1. Introduction")).strip()

    # ---------- discussion ----------
    disc_body = slice_between(intro_disc, "# 2. Discussion", "## References")
    disc_body = strip_first_heading(disc_body, "# 2. Discussion")
    disc_body = fix_crossrefs(disc_body).strip()
    for old, new in [("## 2.1 ", "### 4.1 "), ("## 2.2 ", "### 4.2 "),
                     ("## 2.3 ", "### 4.3 "), ("## 2.4 ", "### 4.4 "),
                     ("## 2.5 ", "### 4.5 ")]:
        assert old in disc_body, f"missing discussion section {old}"
        disc_body = disc_body.replace(old, new)

    # ---------- references ----------
    refs_body = slice_between(intro_disc, "## References", "*Bibliographic details verified by authors")
    refs_body = strip_first_heading(refs_body, "## References").strip()

    # ---------- methods ----------
    m_body = slice_between(methods, "# 1. Cohort construction", "")
    m_body = strip_first_heading(m_body, "# 1. Cohort construction")
    m_body = re.sub(r"^# (\d+)\. ", r"### \1. ", m_body, flags=re.M)
    m_body = fix_crossrefs(m_body).strip()

    # ---------- results ----------
    r1 = slice_between(results, "## Result 1", "## Result 2").strip()
    r1 = strip_first_heading(r1, "## Result 1").strip()
    r2 = slice_between(results, "## Result 2", "## What we do not claim").strip()
    r2 = strip_first_heading(r2, "## Result 2").strip()
    r3 = slice_between(results, "## What we do not claim", "## Controls already in hand").strip()
    r3 = strip_first_heading(r3, "## What we do not claim").strip()

    figures = {
        "f1": "![**Figure 1.** The interaction-fraction gate fails its own validation. "
              "(A) IF vs classical GBDT AUC across the 22 GDSC2 cohorts (PARP inhibitors in violet, "
              "cytotoxic controls in teal); regression slope shows ρ = −0.85, p = 6.9 × 10⁻⁷. "
              "(B) strip plot: all 22 cohorts clear the pre-registered 0.40 gate while arm separation "
              "fails in the wrong direction. Carries Result 1.]"
              "(../figures/fig1_gate.png)",
        "f2": "![**Figure 2.** Encoding scale controls Gram-matrix degeneracy. "
              "(A) mean kernel off-diagonal mass on a log axis: the naive arm (scale = π) collapses to "
              "9.7 × 10⁻⁴ — numerically the identity — while the tuned arm (π/16) sits at 0.65. "
              "(B) per-drug QSVM − SVM-RBF AUC: retuning recovers +0.058 AUC on average, the repair of a "
              "degenerate encoding rather than quantum signal. Carries Result 2.]"
              "(../figures/fig2_encoding.png)",
        "f3": "![**Figure 3.** Per-drug QSVM − SVM-RBF AUC differences with fold-based 95% CIs, both "
              "encoding arms side by side, annotated with Holm-adjusted p. Only the naive arm contains "
              "Holm-significant effects (3/5 drugs), all favouring the classical kernel; the tuned arm "
              "has none. Carries §3.3.]"
              "(../figures/fig3_forest.png)",
    }

    manuscript = f"""# {TITLE}

|  |  |
|---|---|
| **Target venue** | ML4H 2026 Findings (4 pages, non-archival) — primary · ICBINB-BIO, 8 pages — secondary |
| **Evidence level** | `simulated-poc` — exact statevector simulation; no QPU execution |
| **Authors** | _to be completed_ |
| **Status** | Assembled 2026-08-17 from four verified component drafts (see provenance note at end) |

---

## Abstract

{abst_body}

**Short form (120 words).** {short_body}

---

## 1. Introduction

{intro_body}

---

## 2. Methods

> **Evidence level:** `simulated-poc` — all quantum results are exact statevector simulation. No QPU
> was used and no claim in this paper depends on hardware behaviour. Every parameter below was read out
> of the committed scripts and provenance records rather than recalled; the Reproduction section gives
> the exact commands.

{m_body}

---

## 3. Results

### 3.1 Result 1 — the admission gate has no construct validity

{r1}

{figures['f1']}

### 3.2 Result 2 — encoding scale is a hidden hyperparameter that dominates the quantum arm

{r2}

{figures['f2']}

### 3.3 What we do not claim

{r3}

{figures['f3']}

---

## 4. Discussion

{disc_body}

---

## References

{refs_body}

---

<!--
Provenance: assembled 2026-08-17 by scripts/assemble_manuscript.py from
docs/abstract.md, docs/intro_discussion_gate_refutation.md, docs/methods_gate_refutation.md,
docs/paper_draft_gate_refutation.md. Component-level verification at draft time:
verify_abstract.py (22 claims), methods numeric assertions (28), intro/discussion numeric
assertions (18); figure values asserted against source JSON at render time.
-->
"""

    OUT.write_text(manuscript, encoding="utf-8")

    body_only = re.sub(r"<!--.*?-->", "", manuscript, flags=re.S)
    words = len(re.findall(r"\S+", body_only))
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"words: {words}")
    return 0


if __name__ == "__main__":
    sys.exit(main())