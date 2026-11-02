#!/usr/bin/env python3
"""E1 leakage-safe PC1 controls on the frozen 20-drug D3 panel."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
ROOT=Path(__file__).resolve().parent.parent
SEEDS=(42,314,2718); FOLDS=5; RNG=np.random.default_rng(20261003)
def pc_auc(X, y, seed, rank=False, random_direction=False):
 out=[]; ev=[]; tv=[]; corr=[]
 for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
  if rank:
   q=QuantileTransformer(output_distribution="normal",random_state=seed,n_quantiles=min(200,len(tr))).fit(X[tr])
   a=q.transform(X[tr]); b=q.transform(X[te])
  else:
   sc=StandardScaler().fit(X[tr]); a=sc.transform(X[tr]); b=sc.transform(X[te])
  p=PCA(n_components=min(10,a.shape[1]),random_state=seed).fit(a)
  ztr=p.transform(a); zte=p.transform(b)
  if random_direction:
   v=np.random.default_rng(seed + len(tr)).normal(size=a.shape[1]); v/=np.linalg.norm(v)
   train_direction=a@v; test_direction=b@v
  else:
   train_direction=ztr[:,0]; test_direction=zte[:,0]
  clf=LogisticRegression(max_iter=2000).fit(train_direction.reshape(-1,1),y[tr])
  prob=clf.predict_proba(test_direction.reshape(-1,1))[:,1]
  out.append(roc_auc_score(y[te],prob)); ev.append(p.explained_variance_ratio_[0]); tv.append(float(np.var(test_direction))); corr.append(float(spearmanr(train_direction,y[tr]).statistic))
 return float(np.mean(out)),float(np.mean(ev)),float(np.mean(tv)),float(np.mean(corr))
def main():
 out=ROOT/'benchmarks/results';man=json.load(open(out/'icbinb_d3_frozen_selection_20drug.json')); rows=[]
 tuned={r['drug']:r for r in csv.DictReader(open(out/'icbinb_d3_frozen_pi16_3seed_tuned_classical_per_drug.csv'))}
 c1={r['drug']:r for r in json.load(open(out/'icbinb_c1_shap_convention_IF.json'))['cohorts']}
 for item in man['selected']:
  d=item['drug']; base=ROOT/f'data/gdsc_{d}_quantile';X=np.load(base/'X.npy').astype(float);y=np.load(base/'y.npy').astype(int)
  vals=[pc_auc(X,y,s) for s in SEEDS]; rankvals=[pc_auc(X,y,s,rank=True) for s in SEEDS];randvals=[pc_auc(X,y,s,random_direction=True) for s in SEEDS]
  pcauc,ev,tv,cor=np.mean(vals,axis=0); ra=np.mean(rankvals,axis=0)[0]; rnd=np.mean(randvals,axis=0)[0]
  # feature-wise permutation null: permute each feature independently, PCA fit, one deterministic null per seed
  null=[]
  for s in SEEDS:
   rr=np.random.default_rng(s); xp=np.column_stack([rr.permutation(X[:,j]) for j in range(X.shape[1])]);
   null.append(PCA(n_components=1).fit(StandardScaler().fit_transform(xp)).explained_variance_ratio_[0])
  flag=bool(ev>np.quantile(null,.95))
  r=tuned[d];q=float(r['qsvm_auc']);g=float(r['tuned_gbdt_auc']);
  rows.append({'drug':d,'class':item['class'],'n':len(y),'pc1_auc':pcauc,'pc1_explained_variance_train':ev,'pc1_test_projection_variance':tv,'pc1_response_spearman_train':cor,'pc1_flag_perm95':flag,'pc1_null_mean':float(np.mean(null)),'rank_gaussian_pc1_auc':ra,'random_orthogonal_auc':rnd,'if_full':float(c1[d]['if_full']),'if_gate_full_040':float(c1[d]['if_full'])>=.4,'qsvm_auc':q,'tuned_gbdt_auc':g,'qsvm_minus_gbdt':q-g})
 fields=list(rows[0]);
 with open(out/'icbinb_e1_pc1_per_drug.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 def assoc(xs,ys):
  z=spearmanr(xs,ys);return {'rho':float(z.statistic),'p':float(z.pvalue)}
 x=[r['pc1_explained_variance_train'] for r in rows]
 summary={'format':'icbinb_e1_leakage_safe_pc1.v1','design':'PCA and preprocessing fit only in each outer training split; held-out projected with train parameters. QK/tuned GBDT are reused from frozen D3 artifacts.','n_drugs':20,'seeds':list(SEEDS),'per_drug_csv':'benchmarks/results/icbinb_e1_pc1_per_drug.csv','flags':{'pc1_perm95_count':sum(r['pc1_flag_perm95'] for r in rows),'if_gate_count':sum(r['if_gate_full_040'] for r in rows),'intersection':sum(r['pc1_flag_perm95'] and r['if_gate_full_040'] for r in rows),'union':sum(r['pc1_flag_perm95'] or r['if_gate_full_040'] for r in rows)},'associations':{'pc1ev_vs_if':assoc(x,[r['if_full'] for r in rows]),'pc1ev_vs_tuned_gbdt':assoc(x,[r['tuned_gbdt_auc'] for r in rows]),'pc1ev_vs_qk_minus_gbdt':assoc(x,[r['qsvm_minus_gbdt'] for r in rows])},'negative_controls':'rank/Gaussianized PC1 and random orthogonal one-dimensional projection recorded per drug. Tissue-within centering not performed: available matrix row IDs can be joined to model list, but tissue-centering would need a pre-specified handling for sparse tissues; reported as limitation.'}
 json.dump(summary,open(out/'icbinb_e1_pc1_summary.json','w'),indent=2);print('wrote E1',summary['flags'])
if __name__=='__main__':main()
