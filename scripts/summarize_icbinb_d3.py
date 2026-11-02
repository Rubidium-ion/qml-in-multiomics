#!/usr/bin/env python3
"""Aggregate the frozen D3 QK benchmark without treating folds as drug replicates."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parent.parent
M=ROOT/'benchmarks/results/icbinb_d3_frozen_selection_20drug.json'
RUNS=ROOT/'benchmarks/results/icbinb_d3_runs'
OUTJ=ROOT/'benchmarks/results/icbinb_d3_frozen_pi16_3seed_summary.json'
OUTC=ROOT/'benchmarks/results/icbinb_d3_frozen_pi16_3seed_per_drug.csv'
SEEDS=(42,314,2718); SCALE=np.pi/16; NBOOT=10000; NPERM=100000; RNG=np.random.default_rng(20261002)

def q(a,x): return float(np.quantile(a,x))
def hl(d):
    # one-sample HL location: median of Walsh averages
    return float(np.median((d[:,None]+d[None,:]).ravel()/2))
def signflip_mean(d):
    obs=float(d.mean()); signs=RNG.choice((-1.,1.), size=(NPERM,len(d)))
    null=(signs*d).mean(axis=1)
    return {'statistic':'mean drug-level delta AUC','observed':obs,'n_permutations':NPERM,
            'p_two_sided':float((np.sum(np.abs(null)>=abs(obs))+1)/(NPERM+1)),
            'null_q025':q(null,.025),'null_q975':q(null,.975)}
def bootstrap_ci(d):
    med=np.empty(NBOOT); h=np.empty(NBOOT)
    for i in range(NBOOT):
        s=d[RNG.integers(0,len(d),len(d))]
        med[i]=np.median(s); h[i]=hl(s)
    return {'n_boot':NBOOT,'median_ci95':[q(med,.025),q(med,.975)],
            'hodges_lehmann':hl(d),'hodges_lehmann_bootstrap_ci95':[q(h,.025),q(h,.975)]}
def cluster_boot(d, drugs, comp):
    blocks=[]
    for c in comp:
        vals=[d[drugs.index(x)] for x in c if x in drugs]
        if vals: blocks.append(np.array(vals))
    obs=float(np.median(d))
    if len(blocks)<2:
        return {'n_blocks':len(blocks),'observed_median':obs,'ci95':[obs,obs],
                'note':'Degenerate by design: only one overlap component available.'}
    a=[]
    for _ in range(NBOOT):
        take=RNG.integers(0,len(blocks),len(blocks))
        a.append(float(np.median(np.concatenate([blocks[j] for j in take]))))
    return {'n_blocks':len(blocks),'observed_median':obs,'ci95':[q(a,.025),q(a,.975)],
            'n_boot':NBOOT,'blocks':comp,
            'note':'Resamples J>=0.8 overlap components, not drugs independently.'}

def main():
    man=json.load(open(M)); selected=[x['drug'] for x in man['selected']]
    rows=[]
    for drug in selected:
        reps=[]
        for seed in SEEDS:
            p=RUNS/f'{drug}_seed{seed}_pi16.json'
            if not p.exists(): raise SystemExit(f'missing {p}')
            r=json.load(open(p))
            if not np.isclose(r['encoding_scale'],SCALE): raise SystemExit(f'wrong scale {p}')
            reps.append(r)
        def vals(model,key): return np.array([r['models'][model][key] for r in reps],float)
        qauc,rauc,gauc=vals('QSVM','auc'),vals('SVM-RBF','auc'),vals('GBDT','auc')
        qb,rb,gb=vals('QSVM','bal_acc'),vals('SVM-RBF','bal_acc'),vals('GBDT','bal_acc')
        rows.append({'drug':drug,'class':next(x['class'] for x in man['selected'] if x['drug']==drug),
          'n':reps[0]['n_samples'],'n_seeds':3,'n_outer_folds':5,'scale':float(SCALE),
          'kernel_offdiag_mean':float(np.mean([r['kernel_offdiag_mean'] for r in reps])),
          'qsvm_auc':float(qauc.mean()),'svm_rbf_auc':float(rauc.mean()),'gbdt_auc':float(gauc.mean()),
          'delta_qsvm_minus_rbf_auc':float((qauc-rauc).mean()),'delta_qsvm_minus_gbdt_auc':float((qauc-gauc).mean()),
          'qsvm_balacc':float(qb.mean()),'svm_rbf_balacc':float(rb.mean()),'gbdt_balacc':float(gb.mean()),
          'delta_qsvm_minus_rbf_balacc':float((qb-rb).mean()),'delta_qsvm_minus_gbdt_balacc':float((qb-gb).mean())})
    with open(OUTC,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    overlap=json.load(open(ROOT/'benchmarks/results/icbinb_d5d6_cellline_overlap.json'))
    comps=overlap['components']['primary_J_ge_0.8']['components']
    def analysis(key):
        d=np.array([r[key] for r in rows]); drugs=[r['drug'] for r in rows]
        wil=stats.wilcoxon(d,alternative='two-sided',method='auto')
        # frozen practical margin: support for a median >= .05 is assessed against .05.
        wm=stats.wilcoxon(d-.05,alternative='less',method='auto')
        return {'metric':'AUC (already normalized to [0,1])','comparison':key,
          'n_drugs':len(d),'median_delta':float(np.median(d)),'mean_delta':float(d.mean()),
          'n_favouring_qsvm':int((d>0).sum()),'range':[float(d.min()),float(d.max())],
          'wilcoxon_two_sided':{'statistic':float(wil.statistic),'p':float(wil.pvalue)},
          'wilcoxon_margin_0_05_alternative_median_less':{'statistic':float(wm.statistic),'p':float(wm.pvalue),
             'interpretation':'small p supports median delta < +0.05; this is not evidence of quantum advantage.'},
          'signflip_permutation':signflip_mean(d),'effect_interval':bootstrap_ci(d),
          'cluster_aware_bootstrap':cluster_boot(d,drugs,comps)}
    report={'format':'icbinb_revision_d3_frozen_pi16_3seed_summary.v1','selection_manifest':str(M.relative_to(ROOT)),
      'selection_sha256':man['sha256'],'selection_manifest_precedes_quantum_runs':True,
      'n_drugs':20,'seeds':list(SEEDS),'outer_cv':'StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), shared within each drug/seed across QK and controls',
      'quantum':'exact statevector ZZFeatureMap(12 qubits, reps=2, linear)',
      'fixed_encoding_scale':float(SCALE),'feature_selection':'GBDT importance refit on each outer training fold',
      'classical_controls':'SVM-RBF and GBDT on identically selected/scaled outer-fold features; controls use fixed default hyperparameters (no inner-CV tuning in this execution).',
      'unit_of_cross_drug_inference':'drug-level mean across the 3 seeds; folds quantify within-drug spread and are not treated as independent drugs.',
      'per_drug_csv':str(OUTC.relative_to(ROOT)),'analyses':{
         'qsvm_minus_svm_rbf':analysis('delta_qsvm_minus_rbf_auc'),
         'qsvm_minus_gbdt':analysis('delta_qsvm_minus_gbdt_auc')},
      'limitations':['Primary D3 design called for fold-local inner-CV for classical hyperparameters. This completed run uses fixed classical defaults; it is a matched fixed-baseline benchmark, not a fully inner-tuned classical comparison.',
                     'At J>=0.8 the D5/D6 overlap graph has only two components (17+3); cluster bootstrap is necessarily coarse and must be interpreted with the overlap analysis.']}
    report['sha256']=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    json.dump(report,open(OUTJ,'w'),indent=2)
    print('WROTE',OUTJ); print('WROTE',OUTC)
    for k,v in report['analyses'].items():
      print(k, 'median',v['median_delta'],'HL',v['effect_interval']['hodges_lehmann'],'wilcoxon',v['wilcoxon_two_sided']['p'],'margin_less',v['wilcoxon_margin_0_05_alternative_median_less']['p'])
if __name__=='__main__': main()
