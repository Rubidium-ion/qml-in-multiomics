#!/usr/bin/env python
"""Protocol v1.2 -- 14-gene cross-module exact quantum kernel (Gold Standard).

Fixed gene panel from bio (no SHAP top-k selection, so no selection leakage
and no p-hacking surface). Exact Statevector fidelity kernel at 14 qubits.

Efficiency: one statevector per SAMPLE (O(n)), Gram via BLAS -- NOT one
circuit per pair (O(n^2)), which is what made earlier attempts intractable.

Statistics: DeLong test (QSVM vs GBDT) + 95% bootstrap CI on delta AUC,
computed on pooled out-of-fold predictions.
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

PANEL = ["MYOF", "THBS1", "TGFBI", "BRCA1", "RAD51", "SHLD1", "SHLD2",
         "POLQ", "LIG4", "ATM", "FANCD2", "CHEK1", "PARP1", "PALB2"]
MODULE = {"MYOF": "EMT", "THBS1": "EMT", "TGFBI": "EMT-reg",
          "BRCA1": "HR", "RAD51": "HR", "PALB2": "HR",
          "SHLD1": "Shieldin", "SHLD2": "Shieldin",
          "POLQ": "alt-EJ", "LIG4": "NHEJ",
          "ATM": "DDR", "FANCD2": "FA", "CHEK1": "Cycle", "PARP1": "target"}


# ---- DeLong (Sun & Xu 2014 fast algorithm) ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, float); out[J] = T
    return out


def delong(y, p1, p2):
    """Two-sided DeLong test for two correlated ROC AUCs."""
    order = np.argsort(-y, kind="mergesort")
    y = y[order]; p1 = p1[order]; p2 = p2[order]
    m = int(y.sum()); n = len(y) - m
    preds = np.vstack([p1, p2])
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    s = np.atleast_2d(s)
    L = np.array([[1, -1]])
    var = float(L @ s @ L.T)
    if var <= 0:
        return float(auc[0]), float(auc[1]), float("nan"), 1.0
    z = float((auc[0] - auc[1]) / np.sqrt(var))
    from scipy.stats import norm
    return float(auc[0]), float(auc[1]), z, float(2 * (1 - norm.cdf(abs(z))))


def boot_ci(y, pa, pb, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    d = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(d.mean())


def run(cohort, scale, reps, folds, seed, tag):
    root = Path(cohort)
    X = np.load(root / "X.npy"); y = np.load(root / "y.npy")
    names = json.load(open(root / "feature_names.json"))["gene_names"]
    drug = json.load(open(root / "provenance.json"))["drug"]
    idx = [names.index(g) for g in PANEL]
    Xp = X[:, idx]

    fm = ZZFeatureMap(feature_dimension=14, reps=reps, entanglement="linear")
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = {k: np.zeros(len(y)) for k in ("QSVM", "GBDT", "SVM-R", "MLP")}
    offd, t0 = [], time.time()

    for f, (tr, te) in enumerate(skf.split(Xp, y), 1):
        std = StandardScaler().fit(Xp[tr])
        Ztr, Zte = std.transform(Xp[tr]), std.transform(Xp[te])
        mm = MinMaxScaler((0, scale)).fit(Ztr)
        Qtr = mm.transform(Ztr); Qte = np.clip(mm.transform(Zte), 0, scale)

        ptr = np.array([Statevector(fm.assign_parameters(x)).data for x in Qtr])
        pte = np.array([Statevector(fm.assign_parameters(x)).data for x in Qte])
        Ktr = np.abs(ptr.conj() @ ptr.T) ** 2
        Kte = np.abs(pte.conj() @ ptr.T) ** 2
        offd.append(float(Ktr[~np.eye(len(Ktr), dtype=bool)].mean()))
        oof["QSVM"][te] = SVC(kernel="precomputed", C=1.0).fit(
            Ktr, y[tr]).decision_function(Kte)

        oof["GBDT"][te] = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=seed
        ).fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
        oof["SVM-R"][te] = SVC(kernel="rbf", C=1.0, gamma="scale").fit(
            Ztr, y[tr]).decision_function(Zte)
        oof["MLP"][te] = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1500, random_state=seed
        ).fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
        print(f"  [{tag}] fold {f}/{folds} done", flush=True)

    res = {m: {"auc": float(roc_auc_score(y, p)),
               "bal_acc": float(balanced_accuracy_score(y, (p > np.median(p)).astype(int)))}
           for m, p in oof.items()}
    a1, a2, z, p_de = delong(y.astype(float), oof["QSVM"], oof["GBDT"])
    lo, hi, mean_d = boot_ci(y, oof["QSVM"], oof["GBDT"])
    d_gbdt = res["QSVM"]["auc"] - res["GBDT"]["auc"]
    verdict = ("QUANTUM ADVANTAGE" if d_gbdt > 0.02 and p_de < 0.05 and lo > 0
               else "NEGATIVE -- no quantum advantage")
    return {
        "drug": drug, "protocol": "v1.2 (14-gene cross-module, exact statevector)",
        "panel": PANEL, "modules": MODULE, "n_qubits": 14, "reps": reps,
        "encoding_scale": scale, "hilbert_dim": 2 ** 14,
        "kernel_offdiag_mean": float(np.mean(offd)),
        "n_samples": int(len(y)), "folds": folds,
        "feature_selection": "FIXED bio panel -- no SHAP selection, no leakage",
        "models": res,
        "delta_auc_vs_gbdt": float(d_gbdt),
        "delta_auc_vs_svmr": float(res["QSVM"]["auc"] - res["SVM-R"]["auc"]),
        "delong": {"auc_qsvm": a1, "auc_gbdt": a2, "z": z, "p_value": p_de},
        "bootstrap_ci95": {"lo": lo, "hi": hi, "mean_delta": mean_d,
                           "excludes_zero": bool(lo > 0)},
        "success_gate": {"delta_gt_0.02": bool(d_gbdt > 0.02),
                         "delong_p_lt_0.05": bool(p_de < 0.05),
                         "ci_lower_gt_0": bool(lo > 0)},
        "verdict": verdict,
        "wall_clock_sec": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--scale", type=float, default=np.pi)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.cohort, a.scale, a.reps, a.folds, a.seed, a.tag or a.out)
    json.dump(r, open(a.out, "w"), indent=2)
    print(f"\n{r['drug']} scale={a.scale:.4f} reps={a.reps}")
    for m, v in r["models"].items():
        print(f"  {m:<7} AUC {v['auc']:.4f}")
    print(f"  dAUC vs GBDT {r['delta_auc_vs_gbdt']:+.4f}  "
          f"DeLong p={r['delong']['p_value']:.4f}  "
          f"CI95 [{r['bootstrap_ci95']['lo']:+.4f},{r['bootstrap_ci95']['hi']:+.4f}]")
    print(f"  offdiag={r['kernel_offdiag_mean']:.6f}  -> {r['verdict']}")
    print(f"wrote {a.out}")
