#!/usr/bin/env python3
"""pca_ablation.py — how much signal survives PCA-to-n_qubits?

Runs classical models on (a) full 46-D features and (b) the exact same
PCA-reduced space the VQC sees. If classical accuracy also collapses at 8-D,
the bottleneck is dimensionality reduction, not the quantum model.

authors 2026-07-31.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

D = os.path.join(REPO, "data", "tcga_brca_v1")
Xtr, Xte = np.load(f"{D}/X_train.npy"), np.load(f"{D}/X_test.npy")
ytr, yte = np.load(f"{D}/y_train.npy"), np.load(f"{D}/y_test.npy")


def prep(n_comp):
    steps = [("var", VarianceThreshold(1e-8)), ("std", StandardScaler())]
    if n_comp:
        steps += [("pca", PCA(n_components=n_comp, random_state=42)),
                  ("ang", MinMaxScaler((-np.pi / 2, np.pi / 2)))]
    return Pipeline(steps)


MODELS = {
    "svm_rbf": lambda: SVC(kernel="rbf", gamma="scale", probability=True,
                           random_state=42),
    "logreg": lambda: LogisticRegression(max_iter=2000, random_state=42),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000,
                                 random_state=42),
}

rows = []
for n_comp in [None, 8, 4]:
    p = prep(n_comp)
    A, B = p.fit_transform(Xtr), p.transform(Xte)
    evr = (float(p.named_steps["pca"].explained_variance_ratio_.sum())
           if n_comp else 1.0)
    for name, ctor in MODELS.items():
        m = ctor().fit(A, ytr)
        yp, pr = m.predict(B), m.predict_proba(B)
        rows.append({
            "dims": n_comp or int(A.shape[1]), "model": name,
            "explained_var": round(evr, 4),
            "accuracy": round(float(accuracy_score(yte, yp)), 4),
            "f1": round(float(f1_score(yte, yp, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(yte, pr[:, 1])), 4),
        })
        print(f"dims={rows[-1]['dims']:>2} evr={evr:.3f} {name:<8} "
              f"acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['auc_roc']:.3f}")

out = os.path.join(REPO, "benchmarks", "results", "pca_ablation_001.json")
json.dump({"rows": rows,
           "majority_baseline": float(max(np.bincount(yte)) / len(yte)),
           "n_train": int(len(ytr)), "n_test": int(len(yte))},
          open(out, "w"), indent=2)
print("\nwrote", out)
