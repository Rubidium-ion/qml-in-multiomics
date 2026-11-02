"""d5d6 — cross-drug non-independence statistics (Jaccard overlap + block bootstrap).

This module is the reusable core for ICBINB revision items D5/D6 (cohort
non-independence) and is intentionally **generic**: it operates on

  * a set of named entities (here: drugs) each carrying a set of shared
    atomic units (here: cell-line Sanger IDs), and
  * a 1-D vector of one numeric effect value per entity (here: the paired
    cross-drug effect delta = normalized_metric(QK) - normalized_metric(classical)),

so the same code works for the 20-drug frozen benchmark, the 22-cohort IF
frame, or any future drug/assay panel.

What it gives you
-----------------
``jaccard_matrix``            pairwise Jaccard similarity between the unit sets.
``connected_components``      overlap graph components at a threshold (>=0.8 with
                              sensitivity 0.6/0.9 in D5/D6).
``maximum_low_overlap_subset`` deterministic maximum subset with pairwise J<thr
                              (exact, 2**n search; n<=~22 is fine).
``cluster_bootstrap``         the reusable block/cluster bootstrap: resample
                              overlap components with replacement and evaluate a
                              user statistic on the pooled effect values. This is
                              the workhorse for the non-independent-observation
                              95% CI.
``block_constrained_permutation`` block-aware null distributions (sign-flip or
                              drug-label permutation) for the paired effect test,
                              preserving within-block correlation.

All functions are deterministic for a given ``seed`` (numpy Generator).
Only numpy + stdlib are required.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Jaccard / graph helpers
# ---------------------------------------------------------------------------

def jaccard_matrix(unit_sets: dict) -> tuple[np.ndarray, list]:
    """Pairwise Jaccard similarity over a dict {name: iterable-of-unique-units}.

    Returns (J, names) where ``J`` is an ``n x n`` float matrix (J[i, j] =
    |A_i cap A_j| / |A_i cup A_j|), diagonal 1.0, and ``names`` is the sorted
    key order used to index ``J``.  Empty sets are allowed (they give 0.0
    against any other set, including themselves, except a set vs itself which
    is defined as 1.0 when non-empty and 0.0 when empty).
    """
    names = sorted(unit_sets)
    n = len(names)
    J = np.eye(n, dtype=float)
    for i in range(n):
        si = set(unit_sets[names[i]])
        if not si:
            J[i, i] = 0.0
        for j in range(i + 1, n):
            sj = set(unit_sets[names[j]])
            if not si and not sj:
                v = 0.0
            else:
                v = len(si & sj) / len(si | sj)
            J[i, j] = J[j, i] = v
    return J, names


def _union_find_components(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    comps = list(buckets.values())
    # deterministic order: largest first, then by smallest member
    comps.sort(key=lambda c: (-len(c), min(c)))
    for c in comps:
        c.sort()
    return comps


def connected_components(J: np.ndarray, threshold: float) -> list[list[int]]:
    """Connected components of the graph whose edges are pairs with J >= threshold.

    ``J`` is as returned by :func:`jaccard_matrix`.  Returns a list of
    components (each a sorted list of node indices), ordered largest-first then
    by smallest member index.
    """
    n = J.shape[0]
    edges = [
        (i, j)
        for i, j in itertools.combinations(range(n), 2)
        if J[i, j] >= threshold
    ]
    return _union_find_components(n, edges)


def maximum_low_overlap_subset(J: np.ndarray, threshold: float = 0.5) -> list[int]:
    """Deterministic maximum subset whose every pair satisfies J < threshold.

    This is the maximum independent set of the graph with edges where
    ``J >= threshold``.  For ``n <= 22`` the exhaustive combination search is
    affordable (C(22, 11) ~ 705k).  Ties are broken by the lexicographically
    smallest index tuple, so the result is fully reproducible.
    """
    n = J.shape[0]
    # edge = "conflict" (must NOT both be in the subset)
    conflict = (J >= threshold)
    np.fill_diagonal(conflict, False)

    def is_independent(idx: Sequence[int]) -> bool:
        lst = list(idx)
        for a, b in itertools.combinations(lst, 2):
            if conflict[a, b]:
                return False
        return True

    best = None
    # search from the largest candidate size downward; first valid (lexicographically
    # smallest) at that size is provably maximum.
    for size in range(n, 0, -1):
        for comb in itertools.combinations(range(n), size):
            if is_independent(comb):
                best = list(comb)
                break
        if best is not None:
            break
    return best if best is not None else []


# ---------------------------------------------------------------------------
# Paired cross-drug effect
# ---------------------------------------------------------------------------

def paired_delta(qk: Sequence[float], classical: Sequence[float]) -> np.ndarray:
    """Elementwise paired effect delta = qk - classical (per drug).

    Both inputs must have equal length.  NaN/inf propagate naturally so that a
    missing arm is visible downstream rather than silently dropped.
    """
    a = np.asarray(qk, dtype=float)
    b = np.asarray(classical, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"qk/classical length mismatch: {a.shape} vs {b.shape}")
    return a - b


def _as_blocks(blocks: Sequence[Sequence[int]], n: int) -> list[np.ndarray]:
    blocks = [np.asarray(list(b), dtype=int) for b in blocks]
    flat = np.concatenate([b for b in blocks]) if blocks else np.array([], dtype=int)
    if flat.size != n or set(np.unique(flat)) != set(range(n)):
        raise ValueError("blocks must be a partition of indices 0..n-1")
    return blocks


def _median(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if x.size else float("nan")


def cluster_bootstrap(
    values: Sequence[float],
    blocks: Sequence[Sequence[int]],
    n_boot: int,
    statistic: Callable[[np.ndarray], float] | None = None,
    seed: int | np.random.Generator = 42,
) -> dict:
    """Reusable block/cluster bootstrap for a non-independent effect vector.

    ``values``   : one numeric effect value per entity (length n), e.g. paired
                   deltas.
    ``blocks``   : a partition of indices 0..n-1 into overlap components (from
                   :func:`connected_components`).  Each block is resampled as a
                   unit so within-block correlation is preserved.
    ``n_boot``   : number of bootstrap replicates.
    ``statistic``: callable mapping a 1-D pooled value array -> scalar.
                   Defaults to the median.
    ``seed``     : int or numpy Generator (deterministic).

    Each replicate samples ``len(blocks)`` blocks **with replacement** (each
    block with probability 1/B) and pools the member values in the sampled
    order; the statistic is evaluated on the pooled array.  The distribution of
    replicates brackets the non-independent 95% CI.

    Returns a dict with the observed statistic, the bootstrap percentile CI,
    and the bootstrap standard error.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    blocks = _as_blocks(blocks, n)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    stat = _median if statistic is None else statistic
    n_blocks = len(blocks)
    dist = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        # `chosen` holds block POSITIONS; pool the member values of each sampled
        # block (a block may be drawn more than once, as in block bootstrap).
        pooled = np.concatenate([values[blocks[k]] for k in chosen])
        dist[i] = stat(pooled)
    obs = stat(values)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return {
        "observed": obs,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "se": float(np.nanstd(dist)),
        "n_boot": int(n_boot),
        "n_blocks": int(n_blocks),
    }


def block_constrained_permutation(
    values: Sequence[float],
    blocks: Sequence[Sequence[int]],
    n_perm: int,
    mode: str = "sign_flip",
    statistic: Callable[[np.ndarray], float] | None = None,
    seed: int | np.random.Generator = 42,
) -> dict:
    """Block-constrained permutation / sign-flip null for the paired effect.

    This is the reusable engine behind the D5/D6 ">=10,000 constrained
    permutations" requirement.  Two nulls, both block-aware (so the within-block
    dependence of the effect values is preserved under the null):

    * ``mode="sign_flip"`` (default) — H0: each drug's paired delta is symmetric
      about 0 (no systematic cross-drug effect).  Per replicate, the sign of
      every value in a random block is flipped with probability 1/2; the test
      statistic is the chosen statistic of the signed vector.
    * ``mode="label_perm"`` — H0: drug labels are exchangeable given the
      multiset of effect values (the observed ranking/order is chance).  Per
      replicate, the values are permuted **within the union of the sampled
      blocks** (block membership fixed) before the statistic is evaluated.

    ``statistic`` defaults to the median; ``seed`` int or Generator.

    Returns the observed statistic, the two-sided p-value
    (``(#{|T_null| >= |T_obs|} + 1) / (n_perm + 1)``, guarding NaN), and the
    null distribution quantiles for inspection.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    blocks = _as_blocks(blocks, n)
    if mode not in ("sign_flip", "label_perm"):
        raise ValueError(f"unknown mode {mode!r}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    stat = _median if statistic is None else statistic
    obs = stat(values)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        v = values.copy()
        if mode == "sign_flip":
            for b in blocks:
                if rng.integers(2):
                    v[b] = -v[b]
        else:  # label_perm: shuffle values within each block (block size fixed)
            for b in blocks:
                perm = rng.permutation(b.size)
                v[b] = values[b][perm]
        null[i] = stat(v)
    null = null[np.isfinite(null)]
    obs_f = obs if np.isfinite(obs) else 0.0
    ge = np.sum(np.abs(null) >= abs(obs_f))
    p = float((ge + 1) / (n_perm + 1))
    return {
        "mode": mode,
        "observed": obs,
        "p_value": p,
        "n_perm": int(n_perm),
        "null_median": float(np.median(null)) if null.size else float("nan"),
        "null_p975": float(np.percentile(null, 97.5)) if null.size else float("nan"),
        "null_p025": float(np.percentile(null, 2.5)) if null.size else float("nan"),
    }
