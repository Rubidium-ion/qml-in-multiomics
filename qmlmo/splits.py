"""Cross-validation split strategies, including group-aware ones.

Why this module exists as its own file: the three real cohorts we are staging
each fail under plain StratifiedKFold, for the same underlying reason from three
different directions.

  CPTAC pooled  -> rows grouped by tumour collection. Pancreatic, endometrial and
                   renal CT differ in anatomy and contrast protocol. A random
                   split lets the model classify ORGAN and score well.
  CAMELYON17    -> rows grouped by hospital (5 centres). Stain and scanner
                   response differ per centre and centre correlates with label
                   prevalence, so a random split rewards recognising the scanner.
  OASIS-3       -> rows grouped by subject (longitudinal, many sessions each).
                   A random split puts session 1 in train and session 2 in test
                   for the SAME person, which leaks identity outright.

In all three the honest question is "does this generalise to an unseen
group?", and only a group-aware split asks it.

`leave_one_group_out` is the strict version and is what should be reported for
CAMELYON17 and pooled CPTAC. `grouped` (GroupKFold) is the cheaper version for
when there are many groups, e.g. OASIS-3 subjects.
"""
from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np


class SplitPlan:
    """A named list of (train_idx, test_idx) plus what each fold held out."""

    def __init__(self, kind: str, folds: list[tuple[np.ndarray, np.ndarray]],
                 fold_labels: list[str], meta: dict[str, Any] | None = None) -> None:
        self.kind = kind
        self.folds = folds
        self.fold_labels = fold_labels
        self.meta = meta or {}

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for i, (tr, te) in enumerate(self.folds):
            yield i, tr, te

    def __len__(self) -> int:
        return len(self.folds)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_folds": len(self.folds),
            "fold_labels": self.fold_labels,
            "test_sizes": [int(len(te)) for _tr, te in self.folds],
            **self.meta,
        }


def build_splits(y: np.ndarray, cv: str, n_splits: int, seed: int,
                 groups: Sequence[Any] | None = None) -> SplitPlan:
    """Construct a SplitPlan, refusing silently-wrong configurations."""
    y = np.asarray(y).ravel()
    n = y.shape[0]

    if cv in {"grouped", "leave_one_group_out"}:
        if groups is None:
            raise ValueError(
                f"cv={cv!r} requires group labels. Add `group_column` to the "
                "labels section of the config (e.g. cohort / centre / subject_id). "
                "Running a group-aware cohort with a random split is the exact "
                "failure mode this option exists to prevent."
            )
        g = np.asarray([str(x) for x in groups])
        if g.shape[0] != n:
            raise ValueError(f"{g.shape[0]} group labels for {n} samples")
        uniq = sorted(set(g.tolist()))
        if len(uniq) < 2:
            raise ValueError(
                f"cv={cv!r} needs at least 2 groups, found {uniq}. "
                "With one group there is no held-out group to generalise to."
            )

    if cv == "stratified":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(tr, te) for tr, te in skf.split(np.zeros(n), y)]
        return SplitPlan("stratified", folds,
                         [f"fold{i}" for i in range(len(folds))],
                         {"warning": None})

    if cv == "leave_one_group_out":
        from sklearn.model_selection import LeaveOneGroupOut
        g = np.asarray([str(x) for x in groups])
        logo = LeaveOneGroupOut()
        folds, labels = [], []
        for tr, te in logo.split(np.zeros(n), y, groups=g):
            folds.append((tr, te))
            labels.append(str(g[te][0]))
        order = np.argsort(labels)
        folds = [folds[i] for i in order]
        labels = [labels[i] for i in order]
        return SplitPlan("leave_one_group_out", folds, labels,
                         {"n_groups": len(labels),
                          "group_sizes": {l: int((g == l).sum()) for l in labels}})

    if cv == "grouped":
        from sklearn.model_selection import GroupKFold
        g = np.asarray([str(x) for x in groups])
        k = min(n_splits, len(set(g.tolist())))
        if k < n_splits:
            # not an error: fewer groups than requested folds is common and fine,
            # but it must be visible in the result JSON rather than silent
            pass
        gkf = GroupKFold(n_splits=k)
        folds = [(tr, te) for tr, te in gkf.split(np.zeros(n), y, groups=g)]
        return SplitPlan("grouped", folds,
                         [f"fold{i}" for i in range(len(folds))],
                         {"n_groups": len(set(g.tolist())),
                          "requested_n_splits": n_splits,
                          "effective_n_splits": k})

    raise ValueError(
        f"unknown cv={cv!r}; expected 'stratified', 'grouped' or 'leave_one_group_out'"
    )


def audit_group_label_balance(y: np.ndarray, groups: Sequence[Any]) -> dict[str, Any]:
    """Report per-group label prevalence.

    This is the diagnostic that tells you whether a group-aware split was
    necessary in the first place. If prevalence is flat across groups, group and
    label are independent and a random split was probably harmless. If it varies
    a lot -- CAMELYON17 is the textbook case -- then any random-split number was
    partly measuring group identity.
    """
    y = np.asarray(y).ravel()
    g = np.asarray([str(x) for x in groups])
    out: dict[str, Any] = {"per_group": {}}
    prevs = []
    for name in sorted(set(g.tolist())):
        m = g == name
        prev = float(y[m].mean()) if m.sum() else float("nan")
        prevs.append(prev)
        out["per_group"][name] = {
            "n": int(m.sum()),
            "positive_rate": round(prev, 4),
            "single_class": bool(len(np.unique(y[m])) < 2),
        }
    out["prevalence_spread"] = round(float(np.nanmax(prevs) - np.nanmin(prevs)), 4)
    out["interpretation"] = (
        "prevalence_spread near 0 means group and label are roughly independent; "
        "a large spread means random-split scores were partly group identity"
    )
    single = [k for k, v in out["per_group"].items() if v["single_class"]]
    if single:
        out["single_class_groups"] = single
        out["warning"] = (
            f"groups {single} contain one class only. Under leave-one-group-out "
            "their fold has no AUC, and accuracy there is trivially the base rate."
        )
    return out
