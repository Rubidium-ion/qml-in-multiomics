#!/usr/bin/env python
"""True quantum-kernel SVM (fidelity kernel, exact statevector).

Key efficiency insight: the fidelity kernel K(i,j)=|<psi_i|psi_j>|^2 needs
only ONE statevector per SAMPLE (O(n)), not one circuit per PAIR (O(n^2)).
Build Psi (n x 2^q) once, then Gram = |Psi Psi^dag|^2 via BLAS.
This is what makes an exact quantum kernel tractable here.

Feature selection happens INSIDE each training fold (no leakage).
"""
import argparse, json, time, warnings
from pathlib import Path

import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector


def statevectors(fm, Xs):
    """One statevector per sample -- O(n), the whole point."""
    return np.array([Statevector(fm.assign_parameters(x)).data for x in Xs])


def fidelity_gram(psi_a, psi_b):
    """|<a|b>|^2 for all pairs, via BLAS."""
    return np.abs(psi_a.conj() @ psi_b.T) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--n-qubits", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--entanglement", default="linear")
    ap.add_argument("--scale", type=float, default=None,
                    help="encoding upper bound; default pi. Small values "
                         "counteract exponential kernel concentration.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.cohort)
    X = np.load(root / "X.npy")
    y = np.load(root / "y.npy")
    raw = json.load(open(root / "feature_names.json"))
    # feature_names.json is {"gene_names": [...]} in this repo; tolerate a bare
    # list too. Getting this wrong silently yields 0 features.
    names = raw["gene_names"] if isinstance(raw, dict) else raw
    if not (isinstance(names, list) and names and isinstance(names[0], str)):
        raise SystemExit(f"feature_names.json malformed: {type(names)}")
    prov = json.load(open(root / "provenance.json"))
    drug = prov["drug"]

    wl = [g.strip() for g in open("data/hr_pathway_whitelist.txt")
          if g.strip() and not g.strip().startswith("#")]
    idx = [i for i, n in enumerate(names) if n in set(wl)]
    # Guard: a silently vanishing whitelist is this project's recurring bug
    # (BUG 1, BUG 2). Fail loudly instead of degrading quietly.
    if len(idx) < len(wl):
        missing = sorted(set(wl) - set(names))
        raise SystemExit(
            f"whitelist rescue FAILED: matched {len(idx)}/{len(wl)}. "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''}")
    Xw = X[:, idx]
    wl_names = [names[i] for i in idx]
    print(f"[{drug}] n={len(y)} whitelist={Xw.shape[1]} "
          f"-> {args.n_qubits} qubits ({args.reps} reps, {args.entanglement})")

    fm = ZZFeatureMap(feature_dimension=args.n_qubits,
                      reps=args.reps, entanglement=args.entanglement)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)
    res = {k: [] for k in ("QSVM", "SVM-RBF", "GBDT")}
    picked, offdiag, t_start = [], [], time.time()

    for f, (tr, te) in enumerate(skf.split(Xw, y), 1):
        Xtr_raw, Xte_raw = Xw[tr], Xw[te]
        ytr, yte = y[tr], y[te]

        # --- feature selection INSIDE the fold (no leakage) ---
        sel = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         random_state=args.seed)
        sel.fit(Xtr_raw, ytr)
        top = np.argsort(sel.feature_importances_)[::-1][:args.n_qubits]
        picked.append([wl_names[i] for i in top])

        std = StandardScaler().fit(Xtr_raw[:, top])
        hi = args.scale if args.scale else np.pi
        mm = MinMaxScaler((0, hi)).fit(std.transform(Xtr_raw[:, top]))
        Xtr = mm.transform(std.transform(Xtr_raw[:, top]))
        Xte = mm.transform(std.transform(Xte_raw[:, top]))
        Xte = np.clip(Xte, 0, hi)

        # --- exact quantum kernel ---
        t0 = time.time()
        psi_tr = statevectors(fm, Xtr)
        psi_te = statevectors(fm, Xte)
        K_tr = fidelity_gram(psi_tr, psi_tr)
        _off = K_tr[~np.eye(len(K_tr), dtype=bool)]
        offdiag.append(float(_off.mean()))
        K_te = fidelity_gram(psi_te, psi_tr)
        t_q = time.time() - t0

        q = SVC(kernel="precomputed", C=1.0, probability=False)
        q.fit(K_tr, ytr)
        s = q.decision_function(K_te)
        res["QSVM"].append((roc_auc_score(yte, s),
                            balanced_accuracy_score(yte, q.predict(K_te))))

        # --- classical controls on the SAME features ---
        r = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, ytr)
        sr = r.decision_function(Xte)
        res["SVM-RBF"].append((roc_auc_score(yte, sr),
                               balanced_accuracy_score(yte, r.predict(Xte))))

        g = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       random_state=args.seed).fit(Xtr, ytr)
        sg = g.predict_proba(Xte)[:, 1]
        res["GBDT"].append((roc_auc_score(yte, sg),
                            balanced_accuracy_score(yte, g.predict(Xte))))

        print(f"  fold {f}: QSVM {res['QSVM'][-1][0]:.4f}  "
              f"SVM-RBF {res['SVM-RBF'][-1][0]:.4f}  "
              f"GBDT {res['GBDT'][-1][0]:.4f}   (kernel {t_q:.1f}s)")

    out = {
        "drug": drug, "cohort": str(root), "n_samples": int(len(y)),
        "quantum": True,
        "kernel": (f"fidelity |<psi_i|psi_j>|^2, exact statevector, "
                   f"ZZFeatureMap({args.n_qubits}q, reps={args.reps}, "
                   f"{args.entanglement})"),
        "n_qubits": args.n_qubits, "reps": args.reps,
        "entanglement": args.entanglement,
        "encoding_scale": float(args.scale) if args.scale else float(np.pi),
        "kernel_offdiag_mean": float(np.mean(offdiag)),
        "hilbert_dim": 2 ** args.n_qubits,
        "feature_selection": "GBDT importance, refit inside each training fold",
        "folds": args.folds, "seed": args.seed,
        "wall_clock_sec": round(time.time() - t_start, 1),
        "genes_per_fold": picked,
        "models": {},
    }
    print()
    for k, v in res.items():
        a = np.array(v)
        out["models"][k] = {
            "auc": float(a[:, 0].mean()), "auc_sd": float(a[:, 0].std()),
            "bal_acc": float(a[:, 1].mean()), "bal_acc_sd": float(a[:, 1].std()),
            "quantum": k == "QSVM",
            # per-fold values retained so that a PAIRED test is possible after
            # the fact. Without these, only drug-level means survive and every
            # downstream comparison is forced to be unpaired and underpowered.
            "auc_folds": [float(x) for x in a[:, 0]],
            "bal_acc_folds": [float(x) for x in a[:, 1]],
        }
        print(f"  {k:<9} AUC {a[:,0].mean():.4f} +/- {a[:,0].std():.4f}   "
              f"bACC {a[:,1].mean():.4f}")

    # ---- fold-paired comparison, not a bare mean difference ----------------
    #
    # The previous implementation declared "quantum kernel outperforms RBF"
    # whenever the mean AUC gap exceeded a hard-coded 0.01. That threshold has
    # no statistical content: fold-to-fold AUC sd on these cohorts is ~0.03, so
    # a 0.012 gap is well inside noise, and the rule fired on single drugs with
    # no multiplicity correction across the drugs screened. Corrected 2026-08-09.
    q_f = np.array(out["models"]["QSVM"]["auc_folds"])
    r_f = np.array(out["models"]["SVM-RBF"]["auc_folds"])
    diff = q_f - r_f
    d = float(q_f.mean() - r_f.mean())

    n = len(diff)
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    if n > 1 and se > 0:
        t_stat = float(diff.mean() / se)
        p_val = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        crit = float(stats.t.ppf(0.975, df=n - 1))
        ci = [float(diff.mean() - crit * se), float(diff.mean() + crit * se)]
    else:
        t_stat = p_val = float("nan")
        ci = [float("nan"), float("nan")]

    out["quantum_advantage_vs_rbf"] = d
    out["paired_test"] = {
        "test": "paired two-sided t-test on per-fold AUC difference (QSVM - SVM-RBF)",
        "n_folds": int(n),
        "mean_diff": float(diff.mean()),
        "sd_diff": sd,
        "t": t_stat,
        "p": p_val,
        "ci95": ci,
        "note": ("Folds share training data, so this is not independent-sample "
                 "inference; treat it as a descriptive effect size with spread, "
                 "not as confirmatory evidence. Cross-drug claims require "
                 "scripts/parp_paired_test.py and multiplicity correction."),
    }
    sig = (p_val < 0.05) and not np.isnan(p_val)
    if not sig:
        out["verdict"] = ("no detectable separation from SVM-RBF "
                          f"(delta {d:+.4f}, paired p={p_val:.3f})")
    else:
        out["verdict"] = (("quantum kernel above SVM-RBF" if d > 0 else
                           "quantum kernel below SVM-RBF") +
                          f" (delta {d:+.4f}, paired p={p_val:.3f}, UNCORRECTED"
                          "; folds share training data and this is one cohort "
                          "among several screened. NOT a standalone result: "
                          "run scripts/parp_paired_test.py for the cross-drug "
                          "paired test with Holm correction before citing.")
    print(f"\n  QSVM - SVM-RBF = {d:+.4f}  (paired p={p_val:.3f})"
          f"\n  -> {out['verdict']}")

    dest = Path(args.out) if args.out else root / "true_qsvm.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
