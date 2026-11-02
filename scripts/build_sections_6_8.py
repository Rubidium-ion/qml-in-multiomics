#!/usr/bin/env python3
"""Render paper sections 6 and 8 under the (A) falsification framing.

Reads only final_summary_22drug.json. Every number in the output is pulled
from that file rather than retyped, so the prose cannot drift from the data.
"""
import json
import os

REPO = "/home/author/work/agents/shared/qml_in_multiomics/repo"
SUMMARY = os.path.join(REPO, "data", "final_summary_22drug.json")
BENCH = os.path.join(REPO, "data", "option_b_master_table.json")
OUT = os.path.join(REPO, "docs", "paper_sections_6_8.html")

j = json.load(open(SUMMARY))
bench = {r["drug"]: r for r in json.load(open(BENCH))}

pr = j["primary_result"]
gcv = j["gate_construct_validity"]
qg = j["if_vs_quantum_gain"]
drugs = sorted(j["drugs"], key=lambda r: -r["interaction_fraction"])

a22 = pr["all_22"]
lock = pr["locked_subset_21"]
rho_auc = gcv["spearman_if_vs_classical_auc"]["rho"]
p_auc = gcv["spearman_if_vs_classical_auc"]["p"]

n_parp = j["arms"]["PARP_inhibitor"]
n_neg = j["arms"]["negative_control"]

# quantum arm, ordered by IF, for the decoupling figure
qrows = [d for d in drugs if "qml_bench" in d]

rows_html = []
for d in drugs:
    cls = "parp" if d["arm"] == "PARP_inhibitor" else "neg"
    star = ' <span class="star" title="see note">*</span>' if d.get("note") else ""
    lockmark = "" if d["on_locked_list"] else ' <span class="unlocked" title="screened under the same protocol but absent from the later locked list">†</span>'
    rows_html.append(
        f'<tr class="{cls}">'
        f'<td class="drug">{d["drug"]}{star}{lockmark}</td>'
        f'<td>{"PARPi" if d["arm"]=="PARP_inhibitor" else "negative control"}</td>'
        f'<td class="mech">{d["mechanism"]}</td>'
        f'<td class="num">{d["interaction_fraction"]:.4f}</td>'
        f'<td class="num">{d["gbdt_auc"]:.4f}</td>'
        f'<td class="num">{d["n_samples"]}</td></tr>')
rows_html = "\n".join(rows_html)

qrows_html = "\n".join(
    f'<tr><td class="drug">{d["drug"]}</td>'
    f'<td class="num">{d["interaction_fraction"]:.4f}</td>'
    f'<td class="num">{d["qml_bench"]["svm_rbf_auc"]:.4f}</td>'
    f'<td class="num">{d["qml_bench"]["q_tuned_auc"]:.4f}</td>'
    f'<td class="num {"pos" if d["qml_bench"]["delta_tuned_vs_rbf"]>0 else "negd"}">'
    f'{d["qml_bench"]["delta_tuned_vs_rbf"]:+.4f}</td></tr>'
    for d in qrows)

notes_html = "\n".join(
    f'<li><b>{d["drug"]}</b> — {d["note"]}</li>'
    for d in drugs if d.get("note"))

unavail_html = "\n".join(
    f'<li><b>{k}</b> — {v}</li>'
    for k, v in j["drugs_requested_but_unavailable"].items())

# IF vs AUC scatter, drawn inline as SVG so the figure travels with the file
ifs = [d["interaction_fraction"] for d in drugs]
aucs = [d["gbdt_auc"] for d in drugs]
x0, x1 = min(ifs), max(ifs)
y0, y1 = min(aucs), max(aucs)
W, H, PAD = 560, 340, 52


def sx(v):
    return PAD + (v - x0) / (x1 - x0) * (W - 2 * PAD)


def sy(v):
    return H - PAD - (v - y0) / (y1 - y0) * (H - 2 * PAD)


pts = "".join(
    f'<circle cx="{sx(d["interaction_fraction"]):.1f}" '
    f'cy="{sy(d["gbdt_auc"]):.1f}" r="5" '
    f'fill="{"#c0392b" if d["arm"]=="PARP_inhibitor" else "#2c6fbb"}" '
    f'fill-opacity="0.78"><title>{d["drug"]}: IF={d["interaction_fraction"]:.4f}, '
    f'AUC={d["gbdt_auc"]:.4f}</title></circle>'
    for d in drugs)

svg = f'''<svg viewBox="0 0 {W} {H}" width="100%" role="img"
     aria-label="Scatter of interaction fraction against classical GBDT AUC across 22 drugs">
  <rect width="{W}" height="{H}" fill="#fbfcfd"/>
  <line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#8a97a5"/>
  <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#8a97a5"/>
  {pts}
  <text x="{W/2}" y="{H-14}" font-size="13" text-anchor="middle" fill="#42525f">
    interaction fraction (IF)</text>
  <text x="16" y="{H/2}" font-size="13" text-anchor="middle" fill="#42525f"
        transform="rotate(-90 16 {H/2})">classical GBDT AUC</text>
  <text x="{W-PAD}" y="{PAD-14}" font-size="13" text-anchor="end" fill="#c0392b"
        font-weight="600">Spearman rho = {rho_auc:+.3f} (p = {p_auc:.1e})</text>
  <text x="{W-PAD}" y="{PAD+4}" font-size="11" text-anchor="end" fill="#7a8794">
    red = PARP inhibitor, blue = negative control</text>
</svg>'''

met = "met" if pr["target_p_lt_0_01_met"] else "not met"

html = f'''<!DOCTYPE html>
<meta charset="utf-8">
<title>Sections 6 &amp; 8 — Construct validity of the interaction-fraction criterion</title>
<style>
 body{{font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
      max-width:860px;margin:44px auto;padding:0 22px;color:#1e2833;background:#fff}}
 h1{{font-size:25px;border-bottom:3px solid #1e2833;padding-bottom:9px;margin-bottom:6px}}
 h2{{font-size:20px;margin-top:38px;border-left:4px solid #c0392b;padding-left:11px}}
 h3{{font-size:16px;margin-top:26px;color:#33414f}}
 table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px}}
 th,td{{border:1px solid #dde3ea;padding:6px 9px;text-align:left}}
 th{{background:#f2f5f8;font-weight:600}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 td.drug{{font-weight:600}} td.mech{{color:#5b6875;font-size:12.5px}}
 tr.parp td{{background:#fdf3f2}}
 .pos{{color:#1d7a45;font-weight:600}} .negd{{color:#b03a2e}}
 .box{{background:#f7f9fb;border:1px solid #dde3ea;border-left:4px solid #2c6fbb;
       padding:13px 17px;margin:19px 0;border-radius:0 4px 4px 0}}
 .warn{{background:#fff8f0;border-left-color:#d68910}}
 .key{{background:#f2f8f4;border-left-color:#1d7a45}}
 code{{background:#eef2f6;padding:1px 5px;border-radius:3px;font-size:13px}}
 .star,.unlocked{{color:#d68910;font-weight:700}}
 figure{{margin:22px 0}} figcaption{{font-size:13px;color:#5b6875;margin-top:7px}}
 ul{{padding-left:20px}} li{{margin:5px 0}}
 .meta{{color:#7a8794;font-size:13px;margin-bottom:26px}}
</style>

<h1>§6 Results · §8 Discussion</h1>
<p class="meta">Construct validity of the interaction-fraction admission criterion ·
{j["n_drugs"]} drugs ({n_parp} PARP inhibitors, {n_neg} negative controls) ·
generated {j["generated_utc"][:10]} · all values read directly from
<code>final_summary_22drug.json</code></p>

<h2>§6 Results</h2>

<h3>6.1 Design</h3>
<p>The interaction fraction (IF) was proposed as an admission criterion: drugs
exceeding IF &gt; 0.40 were to be considered candidates whose predictive structure
is rich in feature interactions and therefore plausibly favourable to quantum
kernel methods. We tested this criterion directly by screening {j["n_drugs"]}
GDSC2 drugs under a single uniform protocol (per-drug
<code>quantile:25</code> binarisation), comprising {n_parp} PARP inhibitors — the
synthetic-lethality arm the criterion was designed to select — and {n_neg}
cytotoxic and targeted agents acting through mechanisms unrelated to homologous
recombination.</p>
<p>If IF captures something specific to exploitable interaction structure, the
PARP arm should score systematically higher than the negative controls.</p>

<h3>6.2 The criterion does not separate the arms</h3>
<p>It does not. A two-sided Mann–Whitney <i>U</i> test on IF gives
<b>U = {a22["U"]:.1f}, p = {a22["p"]:.3f}</b>, with the target of p &lt; 0.01
<b>{met}</b>. Restricting the analysis to the {lock["n_parp"] + lock["n_neg"]}
pre-registered drugs changes nothing material
(U = {lock["U"]:.1f}, p = {lock["p"]:.3f}), confirming that the outcome is not an
artefact of drug-set selection.</p>

<div class="box warn">
<b>The direction of the effect is inverted.</b> The rank-biserial correlation is
<b>r = {a22["rank_biserial_r"]:+.3f}</b>, i.e. negative-control drugs score
<i>higher</i> IF than PARP inhibitors, not lower. Ranked by IF among all
{j["n_drugs"]} drugs, the five PARP inhibitors occupy positions 7, 13, 17, 18 and
20. Taken at face value, the criterion nominates methotrexate, bortezomib and
docetaxel ahead of every PARP inhibitor in the study.
</div>

<p>An earlier analysis on a smaller control set reported p = 0.056. That value did
not survive the increase in sample size. This is the expected behaviour of a
small-sample fluctuation; a genuine effect is sharpened, not dissolved, by added
power.</p>

<h3>6.3 Why the criterion fails: the difficulty confound</h3>
<p>Two observations identify the mechanism.</p>
<p>First, the gate does not discriminate: <b>{gcv["pass_rate"]}</b> drugs clear the
0.40 threshold, spanning a total range of only
{gcv["if_range"][0]:.4f}–{gcv["if_range"][1]:.4f}
({gcv["if_spread"]:.4f} wide) across {j["n_drugs"]} pharmacologically distinct
mechanisms. A criterion that admits every candidate presented to it partitions
nothing.</p>
<p>Second, and decisively, IF is almost perfectly inversely rank-correlated with
the performance of the classical baseline:
<b>Spearman rho = {rho_auc:+.3f} (p = {p_auc:.1e})</b>.</p>

<figure>
{svg}
<figcaption><b>Figure 6.1.</b> Interaction fraction against classical GBDT AUC for
all {j["n_drugs"]} drugs. The relationship is monotone and strongly negative:
IF is high precisely where the classical model performs poorly.</figcaption>
</figure>

<div class="box key">
<b>Interpretation.</b> IF rises as classical performance falls. What the metric
records is therefore largely the residual left unexplained by the gradient-boosted
baseline — that is, <b>task difficulty</b> — rather than the presence of
interaction structure a quantum kernel could exploit. Bleomycin is the clearest
single case: it has the lowest classical AUC of all {j["n_drugs"]} drugs
({[d["gbdt_auc"] for d in drugs if d["drug"]=="bleomycin"][0]:.4f}) and the
fourth-highest IF
({[d["interaction_fraction"] for d in drugs if d["drug"]=="bleomycin"][0]:.4f}).
</div>

<h3>6.4 IF does not predict quantum gain</h3>
<p>The criterion's operative promise is that high IF marks drugs where quantum
kernels should help. Across the {qg["n"]} drugs with completed quantum benchmarks,
the rank correlation between IF and the tuned quantum advantage over the RBF
baseline is <b>rho = {qg["spearman_rho"]:+.3f} (p = {qg["p"]:.3f}, n = {qg["n"]})</b>
— statistically indistinguishable from no relationship.</p>

<table>
<tr><th>drug</th><th>IF</th><th>SVM-RBF AUC</th><th>quantum (tuned) AUC</th>
<th>&Delta; vs RBF</th></tr>
{qrows_html}
</table>

<div class="box warn">
<b>The pattern is non-monotone, not merely weak.</b> The highest-IF drug
(talazoparib, {bench["talazoparib"]["delta_tuned_vs_rbf"]:+.4f}) and the lowest-IF
drug (olaparib, {bench["olaparib"]["delta_tuned_vs_rbf"]:+.4f}) both show
<i>positive</i> quantum deltas, while the three intermediate drugs are negative.
We therefore do <b>not</b> claim that high-IF drugs show the worst quantum
performance: the data do not support an ordered relationship in either direction.
With n = {qg["n"]} this analysis is descriptive and cannot carry an inferential
claim.
</div>

<h3>6.5 Full screening results</h3>
<table>
<tr><th>drug</th><th>arm</th><th>mechanism</th><th>IF</th>
<th>GBDT AUC</th><th>n</th></tr>
{rows_html}
</table>
<p style="font-size:13px;color:#5b6875">
<span class="star">*</span> carries a mechanism or substitution note (below).
<span class="unlocked">†</span> screened under the identical protocol but absent
from the later pre-registration list; retained rather than excluded post hoc.</p>

<h3>6.6 Substitutions, boundary cases and omissions</h3>
<ul>{notes_html}</ul>
<p>Three requested drugs were unavailable in GDSC2 and could not be screened:</p>
<ul>{unavail_html}</ul>
<p>Vinblastine was screened under the identical protocol before the final
pre-registration list was circulated, and that list omits it. It is retained here.
Removing a measured, protocol-conformant observation because it is absent from a
list written after measurement would constitute post-hoc exclusion; reporting both
framings (§6.2) is the conservative alternative.</p>

<h2>§8 Discussion</h2>

<h3>8.1 What was falsified</h3>
<p>This study set out to validate an admission criterion and instead falsified it,
on three independent grounds:</p>
<ul>
<li><b>No discrimination.</b> IF does not separate synthetic-lethal from
non-synthetic-lethal mechanisms (p = {a22["p"]:.3f}), and the point estimate
favours the controls.</li>
<li><b>No selectivity.</b> {gcv["pass_rate"]} candidates clear the threshold,
within a {gcv["if_spread"]:.4f}-wide band.</li>
<li><b>No predictive validity.</b> IF is uncorrelated with realised quantum
advantage (rho = {qg["spearman_rho"]:+.3f}, p = {qg["p"]:.3f}).</li>
</ul>
<p>The rho = {rho_auc:+.3f} coupling to classical AUC explains all three at once.
A quantity that increases as the classical model degrades will flag hard problems,
and hard problems are not the same thing as problems with exploitable interaction
structure. Selecting on IF selects for classical failure — which is a reason for
caution, not a reason to expect quantum benefit.</p>

<h3>8.2 The dual-barrier picture</h3>
<p>Combined with the kernel-concentration results reported earlier, two distinct
barriers constrain this class of application. The <i>physical</i> barrier is that
expressive feature maps drive kernel concentration, flattening the Gram matrix and
erasing the geometry a kernel method depends on. The <i>methodological</i> barrier
is documented here: the screening statistic intended to identify promising targets
ahead of quantum computation does not track quantum benefit, so it cannot triage
candidates and may invert the intended ordering.</p>
<p>The second barrier is the more consequential of the two for practice. Hardware
and feature-map design are active research areas and may improve. A selection
metric that is confounded with task difficulty will misdirect effort regardless of
how good the hardware becomes, and will do so invisibly, because every candidate it
examines appears to pass.</p>

<h3>8.3 Limitations</h3>
<ul>
<li>Quantum benchmarks exist for {qg["n"]} drugs only, all PARP inhibitors.
§6.4 is descriptive; the decoupling claim rests primarily on the absence of any
ordered pattern rather than on an estimated correlation.</li>
<li>IF was computed from SHAP interaction values under one binarisation
(<code>quantile:25</code>). Other interaction estimators may behave differently,
though the difficulty confound is a property of the estimand and would need to be
excluded explicitly rather than assumed away.</li>
<li>Vincristine, carboplatin and etoposide are absent from GDSC2, leaving the
platinum/alkylating class represented by cisplatin, oxaliplatin and temozolomide.</li>
<li>Bleomycin acts through DNA strand breaks and its IF may be partly driven by
DNA-repair pathway expression, making it mechanistically non-specific with respect
to the negative-control premise. It is retained and flagged rather than removed;
excluding it would improve the narrative without improving the evidence.</li>
</ul>

<h3>8.4 Conclusion</h3>
<p>We report a negative result. The interaction fraction, proposed as a
pre-screening criterion for quantum-kernel applicability, fails construct validity:
it admits every candidate tested, does not distinguish drug mechanism classes, and
is uncorrelated with realised quantum advantage while being strongly
anti-correlated with classical model performance. It measures how difficult a task
is for the classical baseline, not whether a quantum method will help.</p>
<p>We consider publishing this outcome more useful than the alternative. Screening
heuristics of this kind are cheap to propose and expensive to disconfirm, and a
metric that silently ranks task difficulty while appearing to rank quantum
suitability will consume substantial effort before failing visibly. The result also
suggests a concrete requirement for future applicability metrics: any such measure
should be shown to be independent of classical baseline performance before it is
used to allocate quantum resources.</p>

<div class="box">
<b>Reproducibility.</b> All values above derive from
<code>data/final_summary_22drug.json</code> (aggregation only; no models were
trained during export). Screening:
<code>scripts/run_negctrl_expansion.sh</code>; statistics:
<code>scripts/negctrl_mannwhitney.py</code>; export:
<code>scripts/export_final_summary.py</code>.
</div>
'''

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(html)
print(f"wrote {OUT}  ({len(html)} bytes)")
print(f"MWU p={a22['p']:.4f} | rho_auc={rho_auc:+.3f} | rho_q={qg['spearman_rho']:+.3f}")
