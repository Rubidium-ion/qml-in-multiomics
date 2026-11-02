#!/usr/bin/env python3
"""Nested-CV tuned classical comparators for frozen D3; reuses completed QK runs.

Outer split, whitelist and fold-local GBDT feature selection match
run_true_qsvm.py exactly. For every outer training fold, hyperparameters are
selected only with an inner StratifiedKFold. No QK statevectors are recomputed.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
ROOT=Path(__file__).resolve().parent.parent
SCALE=float(np.pi/16)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--drug',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--inner-folds',type=int,default=3); ap.add_argument('--out',required=True); a=ap.parse_args()
 cohort=ROOT/f'data/gdsc_{a.drug}_quantile'; qpath=ROOT/f'benchmarks/results/icbinb_d3_runs/{a.drug}_seed{a.seed}_pi16.json'
 q=json.load(open(qpath))
 if not np.isclose(q['encoding_scale'],SCALE): raise SystemExit('QK run scale mismatch')
 X=np.load(cohort/'X.npy'); y=np.load(cohort/'y.npy'); raw=json.load(open(cohort/'feature_names.json')); names=raw['gene_names'] if isinstance(raw,dict) else raw
 wl=[s.strip() for s in open(ROOT/'data/hr_pathway_whitelist.txt') if s.strip() and not s.startswith('#')]
 idx=[i for i,n in enumerate(names) if n in set(wl)]
 if len(idx)!=len(wl): raise SystemExit(f'whitelist mismatch {len(idx)}/{len(wl)}')
 X=X[:,idx]; names=[names[i] for i in idx]
 outer=StratifiedKFold(n_splits=a.folds,shuffle=True,random_state=a.seed)
 grids={'SVM-RBF':({'C':[0.1,1.,10.,100.], 'gamma':['scale',.001,.01,.1]}, SVC(kernel='rbf',probability=True,random_state=a.seed)),
        'GBDT':({'n_estimators':[100,200,400], 'max_depth':[1,2,3,4], 'learning_rate':[.03,.1]}, GradientBoostingClassifier(random_state=a.seed))}
 results={n:{'auc_folds':[],'bal_acc_folds':[],'selected_params':[],'inner_best_auc':[]} for n in grids}
 outer_features=[]; t=time.time()
 for fold,(tr,te) in enumerate(outer.split(X,y)):
  # Matches QK protocol: selector trained exclusively on outer train.
  selector=GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=a.seed).fit(X[tr],y[tr])
  top=np.argsort(selector.feature_importances_)[::-1][:12]; outer_features.append([names[i] for i in top])
  std=StandardScaler().fit(X[tr][:,top]); mm=MinMaxScaler((0,SCALE)).fit(std.transform(X[tr][:,top]))
  xtr=mm.transform(std.transform(X[tr][:,top])); xte=np.clip(mm.transform(std.transform(X[te][:,top])),0,SCALE)
  inner=StratifiedKFold(n_splits=a.inner_folds,shuffle=True,random_state=a.seed*100+fold)
  for name,(grid,est) in grids.items():
   gs=GridSearchCV(est,grid,scoring='roc_auc',cv=inner,n_jobs=-1,refit=True,return_train_score=False)
   gs.fit(xtr,y[tr]); prob=gs.predict_proba(xte)[:,1]; pred=gs.predict(xte)
   results[name]['auc_folds'].append(float(roc_auc_score(y[te],prob))); results[name]['bal_acc_folds'].append(float(balanced_accuracy_score(y[te],pred)))
   results[name]['selected_params'].append(gs.best_params_); results[name]['inner_best_auc'].append(float(gs.best_score_))
  print(f'{a.drug} seed={a.seed} fold={fold+1}/{a.folds} done',flush=True)
 out={'format':'icbinb_d3_tuned_classical_outer_nested_cv.v1','drug':a.drug,'n_samples':int(len(y)),'seed':a.seed,'outer_cv':{'n_splits':a.folds,'shuffle':True,'random_state':a.seed},'inner_cv':{'n_splits':a.inner_folds,'shuffle':True,'random_state':'seed*100+outer_fold','scoring':'roc_auc'},'scale':SCALE,'n_qubits_features':12,'feature_selection':'GBDT importance fitted on outer training fold only; identical outer selection protocol to completed QK runs','qk_reused':str(qpath.relative_to(ROOT)),'qk_auc_folds':q['models']['QSVM']['auc_folds'],'outer_features':outer_features,'models':{},'wall_clock_sec':round(time.time()-t,2)}
 for n,r in results.items():
  out['models'][n]={'auc':float(np.mean(r['auc_folds'])),'auc_sd':float(np.std(r['auc_folds'])),'bal_acc':float(np.mean(r['bal_acc_folds'])),'bal_acc_sd':float(np.std(r['bal_acc_folds'])),**r}
 out['qsvm_minus_svm_rbf_auc']=float(np.mean(q['models']['QSVM']['auc_folds'])-out['models']['SVM-RBF']['auc'])
 out['qsvm_minus_gbdt_auc']=float(np.mean(q['models']['QSVM']['auc_folds'])-out['models']['GBDT']['auc'])
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,'w'),indent=2); print('WROTE',a.out)
if __name__=='__main__': main()
