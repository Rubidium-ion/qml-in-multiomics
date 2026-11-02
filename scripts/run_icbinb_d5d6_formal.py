#!/usr/bin/env python3
"""Formal D5/D6 dependence-aware sensitivity analyses for frozen tuned D3."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0,str(ROOT))
from algorithms.d5d6 import cluster_bootstrap, connected_components
RNG=42; NB=10000

def main():
 out=ROOT/'benchmarks/results'; rows=list(csv.DictReader(open(out/'icbinb_d3_frozen_pi16_3seed_tuned_classical_per_drug.csv')))
 overlap=json.load(open(out/'icbinb_d5d6_cellline_overlap.json'))
 # matrix order saved in CSV, use names to align effects
 with open(out/'icbinb_d5d6_jaccard_matrix.csv') as f:
  m=list(csv.reader(f)); names=m[0][1:]; J=np.array([[float(x) for x in r[1:]] for r in m[1:]])
 by={r['drug']:r for r in rows}; effects={
  'qsvm_minus_tuned_svm_rbf':np.array([float(by[n]['delta_qsvm_minus_tuned_svm_rbf_auc']) for n in names]),
  'qsvm_minus_tuned_gbdt':np.array([float(by[n]['delta_qsvm_minus_tuned_gbdt_auc']) for n in names])}
 records=[]; loo=[]; loco=[]
 for label,d in effects.items():
  for threshold in (.6,.8,.9):
   comps=connected_components(J,threshold)
   b=cluster_bootstrap(d,comps,NB,statistic=np.median,seed=RNG)
   records.append({'comparison':label,'jaccard_threshold':threshold,'n_blocks':len(comps),'component_sizes':';'.join(map(str,map(len,comps))),'observed_median':b['observed'],'ci95_low':b['ci95_low'],'ci95_high':b['ci95_high'],'bootstrap_se':b['se'],'n_boot':NB})
   for i,n in enumerate(names): loo.append({'comparison':label,'jaccard_threshold':threshold,'left_out_drug':n,'median_delta_without':float(np.median(np.delete(d,i)))})
   for k,c in enumerate(comps):
    keep=np.array([i for i in range(len(d)) if i not in set(c)])
    loco.append({'comparison':label,'jaccard_threshold':threshold,'left_out_component':k,'component_drugs':';'.join(names[i] for i in c),'n_remaining':len(keep),'median_delta_without':float(np.median(d[keep])) if len(keep) else None})
 # exact low overlap result from existing metadata
 low=overlap['max_low_overlap_subset_J_lt_0.5']
 low_rows=[]
 for label,d in effects.items():
  sel=[names.index(n) for n in low['subset']]
  low_rows.append({'comparison':label,'criterion':'pairwise J<0.5','subset':';'.join(low['subset']),'subset_size':len(sel),'median_delta':float(np.median(d[sel])),'interpretation':'Non-inferential singleton: no pair of drugs meets J<0.5.'})
 for fn,data in [('icbinb_d5d6_block_bootstrap_summary.csv',records),('icbinb_d5d6_leave_one_drug_out.csv',loo),('icbinb_d5d6_leave_one_component_out.csv',loco),('icbinb_d5d6_low_overlap_subset.csv',low_rows)]:
  with open(out/fn,'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 payload={'format':'icbinb_d5d6_formal_block_bootstrap.v1','input':'icbinb_d3_frozen_pi16_3seed_tuned_classical_per_drug.csv','unit':'drug-level mean AUC delta across 3 seeds','primary_jaccard_threshold':.8,'sensitivity_thresholds':[.6,.8,.9],'n_boot':NB,'results':records,'low_overlap_subset':low_rows,'limitations':'At J>=.6 all 20 drugs form one block; J>=.8/.9 gives only 17+3 blocks. J<.5 maximum subset size=1, so low-overlap analysis cannot be inferential. LODO/LOCO are robustness diagnostics.'}
 json.dump(payload,open(out/'icbinb_d5d6_block_bootstrap_summary.json','w'),indent=2)
 print('wrote formal D5/D6 outputs')
if __name__=='__main__':main()
