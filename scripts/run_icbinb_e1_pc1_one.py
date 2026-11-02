#!/usr/bin/env python3
"""One-drug E1 leakage-safe PC1 analysis, checkpointable by design."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
ROOT=Path(__file__).resolve().parent.parent; SEEDS=(42,314,2718)
def eval_control(X,y,seed,mode):
 out=[]; ev=[]; tvar=[]; corr=[]
 for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
  if mode=='rank':
   q=QuantileTransformer(n_quantiles=min(200,len(tr)),output_distribution='normal',random_state=seed).fit(X[tr]);a,b=q.transform(X[tr]),q.transform(X[te])
  else:
   sc=StandardScaler().fit(X[tr]);a,b=sc.transform(X[tr]),sc.transform(X[te])
  # randomized SVD is train-only and makes 5k-gene PCA practical.
  p=PCA(n_components=10,svd_solver='randomized',random_state=seed*10+fold).fit(a)
  ztr,zte=p.transform(a),p.transform(b)
  if mode=='random':
   rng=np.random.default_rng(seed*100+fold);v=rng.normal(size=a.shape[1]);v/=np.linalg.norm(v);u,vte=a@v,b@v
  else:u,vte=ztr[:,0],zte[:,0]
  clf=LogisticRegression(max_iter=1000).fit(u.reshape(-1,1),y[tr]);prob=clf.predict_proba(vte.reshape(-1,1))[:,1]
  out.append(roc_auc_score(y[te],prob));ev.append(p.explained_variance_ratio_[0]);tvar.append(float(np.var(vte)));corr.append(float(spearmanr(u,y[tr]).statistic))
 return [float(np.mean(z)) for z in (out,ev,tvar,corr)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--drug',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 base=ROOT/f'data/gdsc_{a.drug}_quantile';X=np.load(base/'X.npy').astype(np.float32);y=np.load(base/'y.npy').astype(int)
 normal=np.array([eval_control(X,y,s,'normal') for s in SEEDS]);rank=np.array([eval_control(X,y,s,'rank') for s in SEEDS]);random=np.array([eval_control(X,y,s,'random') for s in SEEDS])
 # Feature-wise permutation null: each feature independently shuffled, same train-only PCA procedure.
 null=[]
 for s in SEEDS:
  rng=np.random.default_rng(10000+s);xp=np.empty_like(X)
  for j in range(X.shape[1]): xp[:,j]=rng.permutation(X[:,j])
  null.append(eval_control(xp,y,s,'normal')[1])
 row={'drug':a.drug,'n':int(len(y)),'pc1_auc':float(normal[:,0].mean()),'pc1_auc_seed_sd':float(normal[:,0].std()),'pc1_explained_variance_train':float(normal[:,1].mean()),'pc1_test_projection_variance':float(normal[:,2].mean()),'pc1_response_spearman_train':float(normal[:,3].mean()),'rank_gaussian_pc1_auc':float(rank[:,0].mean()),'random_orthogonal_auc':float(random[:,0].mean()),'pc1_perm_null_values':null,'pc1_perm_null_95':float(np.quantile(null,.95)),'pc1_flag_perm95':bool(normal[:,1].mean()>np.quantile(null,.95)),'seeds':list(SEEDS),'negative_control_note':'Rank/Gaussianized and random orthogonal controls use training-fitted transforms/directions per outer fold.'}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(row,open(a.out,'w'),indent=2);print('WROTE',a.out)
if __name__=='__main__':main()
