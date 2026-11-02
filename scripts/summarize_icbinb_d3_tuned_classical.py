#!/usr/bin/env python3
"""Aggregate frozen D3 QK results against nested-CV tuned classical controls."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent; RUN=ROOT/'benchmarks/results/icbinb_d3_tuned_classical_runs'; QRUN=ROOT/'benchmarks/results/icbinb_d3_runs'; MAN=ROOT/'benchmarks/results/icbinb_d3_frozen_selection_20drug.json'; RNG=np.random.default_rng(20261002); NB=10000; NP=100000

def quant(a,x): return float(np.quantile(a,x))
def hl(d): return float(np.median(((d[:,None]+d[None,:])/2).ravel()))
def bootstrap(d):
 m=[]; h=[]
 for _ in range(NB):
  z=d[RNG.integers(len(d),size=len(d))];m.append(np.median(z));h.append(hl(z))
 return {'n_boot':NB,'median_ci95':[quant(m,.025),quant(m,.975)],'hodges_lehmann':hl(d),'hodges_lehmann_bootstrap_ci95':[quant(h,.025),quant(h,.975)]}
def sf(d):
 null=(RNG.choice([-1.,1.],(NP,len(d)))*d).mean(1); obs=float(d.mean())
 return {'n_permutations':NP,'statistic':'mean drug-level delta AUC','observed':obs,'p_two_sided':float((np.sum(abs(null)>=abs(obs))+1)/(NP+1)),'null_q025':quant(null,.025),'null_q975':quant(null,.975)}
def cb(d,drugs,comps):
 blocks=[np.array([d[drugs.index(x)] for x in c if x in drugs]) for c in comps]; vals=[]
 for _ in range(NB): vals.append(np.median(np.concatenate([blocks[i] for i in RNG.integers(len(blocks),size=len(blocks))])))
 return {'n_blocks':len(blocks),'blocks':comps,'n_boot':NB,'observed_median':float(np.median(d)),'ci95':[quant(vals,.025),quant(vals,.975)],'note':'J>=0.8 overlap-component bootstrap; components, not drugs, resampled.'}
def test(d,drugs,comps):
 w=stats.wilcoxon(d,alternative='two-sided',method='auto'); margin=stats.wilcoxon(d-.05,alternative='less',method='auto')
 return {'metric':'AUC [0,1]','n_drugs':len(d),'median_delta':float(np.median(d)),'mean_delta':float(np.mean(d)),'n_favouring_qsvm':int(np.sum(d>0)),'range':[float(d.min()),float(d.max())],'wilcoxon_two_sided':{'statistic':float(w.statistic),'p':float(w.pvalue)},'wilcoxon_margin_0_05_alternative_median_less':{'statistic':float(margin.statistic),'p':float(margin.pvalue),'interpretation':'Small p supports median QK advantage below +0.05.'},'signflip_permutation':sf(d),'effect_interval':bootstrap(d),'cluster_aware_bootstrap':cb(d,drugs,comps)}
def main():
 man=json.load(open(MAN)); selected=[r['drug'] for r in man['selected']]; seeds=[42,314,2718]; rows=[]; validation=[]
 for drug in selected:
  qauc=[]; rbf=[]; gbdt=[]
  for seed in seeds:
   q=json.load(open(QRUN/f'{drug}_seed{seed}_pi16.json')); t=json.load(open(RUN/f'{drug}_seed{seed}.json'))
   ok=q['genes_per_fold']==t['outer_features']; validation.append({'drug':drug,'seed':seed,'outer_features_match_cached_qk':ok})
   if not ok: raise RuntimeError(f'feature mismatch {drug} {seed}')
   qauc.append(q['models']['QSVM']['auc']); rbf.append(t['models']['SVM-RBF']['auc']); gbdt.append(t['models']['GBDT']['auc'])
  r=next(x for x in man['selected'] if x['drug']==drug)
  rows.append({'drug':drug,'class':r['class'],'n':r['n'],'qsvm_auc':float(np.mean(qauc)),'tuned_svm_rbf_auc':float(np.mean(rbf)),'tuned_gbdt_auc':float(np.mean(gbdt)),'delta_qsvm_minus_tuned_svm_rbf_auc':float(np.mean(np.array(qauc)-rbf)),'delta_qsvm_minus_tuned_gbdt_auc':float(np.mean(np.array(qauc)-gbdt))})
 outcsv=ROOT/'benchmarks/results/icbinb_d3_frozen_pi16_3seed_tuned_classical_per_drug.csv';outjson=ROOT/'benchmarks/results/icbinb_d3_frozen_pi16_3seed_tuned_classical_summary.json'
 with open(outcsv,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 comps=json.load(open(ROOT/'benchmarks/results/icbinb_d5d6_cellline_overlap.json'))['components']['primary_J_ge_0.8']['components']; drugs=[r['drug'] for r in rows]
 d1=np.array([r['delta_qsvm_minus_tuned_svm_rbf_auc'] for r in rows]);d2=np.array([r['delta_qsvm_minus_tuned_gbdt_auc'] for r in rows])
 out={'format':'icbinb_d3_frozen_pi16_3seed_tuned_classical_summary.v1','selection_manifest':str(MAN.relative_to(ROOT)),'selection_sha256':man['sha256'],'n_drugs':20,'seeds':seeds,'quantum_reused':True,'quantum_scale':float(np.pi/16),'nested_classical_cv':{'inner_folds':3,'selection_scope':'outer training partition only','scoring':'roc_auc','svm_grid':{'C':[.1,1,10,100],'gamma':['scale',.001,.01,.1]},'gbdt_grid':{'n_estimators':[100,200,400],'max_depth':[1,2,3,4],'learning_rate':[.03,.1]}},'validation':{'all_60_outer_feature_sets_exactly_match_cached_qk':all(x['outer_features_match_cached_qk'] for x in validation),'details':validation},'per_drug_csv':str(outcsv.relative_to(ROOT)),'analyses':{'qsvm_minus_tuned_svm_rbf':test(d1,drugs,comps),'qsvm_minus_tuned_gbdt':test(d2,drugs,comps)},'interpretation':'This is the confirmatory classical comparator arm for the frozen D3 sample; prior fixed-default classical comparisons are sensitivity analyses only.'}
 out['sha256']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest();json.dump(out,open(outjson,'w'),indent=2)
 print('WROTE',outjson);print('WROTE',outcsv)
 for k,x in out['analyses'].items():print(k,x['median_delta'],x['effect_interval']['hodges_lehmann'],x['wilcoxon_two_sided']['p'],x['wilcoxon_margin_0_05_alternative_median_less']['p'])
if __name__=='__main__':main()
