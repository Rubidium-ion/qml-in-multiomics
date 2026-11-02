"""Synthetic multi-modal generator with a *known* ground truth.

This exists to answer the question the GDSC work could not: when we say a
quantum kernel found nothing, was there anything to find? Here we control that
by construction.

Signal knobs:
  main_effect_frac : share of label variance from additive main effects
  interaction_frac : share from pairwise products (the structure a fidelity
                     kernel is supposed to be good at)
  cross_modal      : if True, interaction terms pair features across different
                     modalities, so single-modality models are provably capped
  n_interaction_pairs : how many pairs actually carry signal. This matters more
                     than it looks -- see the note in `generate()`.

The generator records the true generative weights in meta, which makes it
possible to test whether a task-admission criterion recovers *known* interaction
structure -- exactly the test the interaction_fraction gate could never pass on
real data because ground truth was unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register_modality
from .base import FeatureBlock, ModalityAdapter


@register_modality
class SyntheticMultiOmicsAdapter(ModalityAdapter):
    """params:
      n_samples=300, blocks={'transcriptomics': 40, 'genomics': 20, ...},
      main_effect_frac=0.5, interaction_frac=0.5, cross_modal=True,
      n_informative=8, n_interaction_pairs=None, noise=1.0,
      class_balance=0.5, seed=42
    Returns the FIRST block; use `generate()` for the full multi-block dataset.
    """

    name = "synthetic"

    def load(self) -> FeatureBlock:
        blocks, _y, _truth = self.generate()
        return blocks[0]

    def generate(self) -> tuple[list[FeatureBlock], np.ndarray, dict[str, Any]]:
        p = self.params
        n = int(p.get("n_samples", 300))
        spec: dict[str, int] = dict(p.get("blocks", {"transcriptomics": 40, "genomics": 20}))
        main_frac = float(p.get("main_effect_frac", 0.5))
        inter_frac = float(p.get("interaction_frac", 0.5))
        cross_modal = bool(p.get("cross_modal", True))
        n_inf = int(p.get("n_informative", 8))
        noise = float(p.get("noise", 1.0))
        balance = float(p.get("class_balance", 0.5))
        seed = int(p.get("seed", 42))

        if not spec:
            raise ValueError("synthetic: `blocks` must name at least one modality")
        if main_frac < 0 or inter_frac < 0 or (main_frac + inter_frac) <= 0:
            raise ValueError(
                "synthetic: main/interaction fractions must be non-negative and not both zero"
            )

        rng = np.random.default_rng(seed)
        names = list(spec)
        mats = {m: rng.standard_normal((n, spec[m])) for m in names}

        # global index of every (modality, column) so interactions can cross blocks
        flat: list[tuple[str, int]] = [(m, j) for m in names for j in range(spec[m])]
        total_k = len(flat)
        if n_inf > total_k:
            raise ValueError(
                f"synthetic: n_informative={n_inf} exceeds {total_k} total features"
            )

        if cross_modal and len(names) > 1:
            # spread informative features across modalities as evenly as possible
            picks: list[int] = []
            per = max(1, n_inf // len(names))
            for m in names:
                cand = [i for i, (mm, _) in enumerate(flat) if mm == m]
                picks.extend(
                    rng.choice(cand, size=min(per, len(cand)), replace=False).tolist()
                )
            remaining = [i for i in range(total_k) if i not in set(picks)]
            if len(picks) < n_inf and remaining:
                picks.extend(
                    rng.choice(remaining, size=n_inf - len(picks), replace=False).tolist()
                )
            informative = np.asarray(picks[:n_inf], dtype=int)
        else:
            informative = rng.choice(total_k, size=n_inf, replace=False)

        col = lambda gi: mats[flat[gi][0]][:, flat[gi][1]]  # noqa: E731

        # ---- main effects ----
        main_w = rng.normal(0.0, 1.0, size=n_inf)
        z_main = sum(float(w) * col(int(gi)) for w, gi in zip(main_w, informative))

        # ---- candidate interaction pairs ----
        candidates: list[tuple[int, int]] = []
        for a in range(n_inf):
            for b in range(a + 1, n_inf):
                ga, gb = int(informative[a]), int(informative[b])
                if cross_modal and len(names) > 1 and flat[ga][0] == flat[gb][0]:
                    continue
                candidates.append((ga, gb))
        if not candidates:  # fall back to within-modality pairs
            candidates = [
                (int(informative[a]), int(informative[b]))
                for a in range(n_inf)
                for b in range(a + 1, n_inf)
            ]

        # Concentrate the interaction signal.
        #
        # Summing over all C(n_inf, 2) pairs with independent random signs makes
        # the terms cancel toward a Gaussian by CLT: the composite becomes
        # near-indistinguishable from noise, and NO model -- classical or quantum --
        # can recover structure that averaged itself away. Measured: with 27 pairs
        # an oracle GBDT on the true informative columns reached AUC 0.538 at
        # noise=0.8, i.e. barely above chance, which would have made every quantum
        # arm look "correctly" null for entirely the wrong reason.
        #
        # A small number of strongly weighted pairs is both learnable and a better
        # model of real epistasis, which is sparse.
        default_pairs = max(1, n_inf // 3)
        n_pairs = int(p.get("n_interaction_pairs", default_pairs))
        n_pairs = max(1, min(n_pairs, len(candidates)))
        if n_pairs < len(candidates):
            sel = rng.choice(len(candidates), size=n_pairs, replace=False)
            pairs = [candidates[int(i)] for i in sel]
        else:
            pairs = list(candidates)

        # magnitudes near 1 with random sign: no cancellation-by-averaging
        inter_w = np.abs(rng.normal(1.0, 0.2, size=len(pairs))) * rng.choice(
            [-1.0, 1.0], size=len(pairs)
        )
        # standardise each product term so one heavy-tailed pair cannot dominate
        z_inter = sum(
            float(w) * _unit(col(ga) * col(gb)) for w, (ga, gb) in zip(inter_w, pairs)
        )

        z_main = _unit(z_main)
        z_inter = _unit(z_inter)
        z_signal = _unit(main_frac * z_main + inter_frac * z_inter)
        z = z_signal + noise * rng.standard_normal(n)

        thresh = float(np.quantile(z, 1.0 - balance))
        y = (z > thresh).astype(np.int32)

        blocks = [
            FeatureBlock(
                X=mats[m],
                feature_names=[f"{m[:4]}_{j:04d}" for j in range(spec[m])],
                sample_ids=[f"syn{i:05d}" for i in range(n)],
                modality=m,
                meta={"adapter": self.name, "synthetic": True, "seed": seed},
            )
            for m in names
        ]

        # theoretical ceiling: Bayes-optimal AUC given the noise level, useful as
        # the honest reference instead of 1.0
        snr = 1.0 / noise if noise > 0 else float("inf")
        truth = {
            "seed": seed,
            "n_samples": n,
            "blocks": spec,
            "main_effect_frac": main_frac,
            "interaction_frac": inter_frac,
            "cross_modal": cross_modal,
            "noise": noise,
            "snr": snr,
            "informative_features": [
                f"{flat[int(g)][0]}:{flat[int(g)][1]}" for g in informative
            ],
            "interaction_pairs": [
                [f"{flat[a][0]}:{flat[a][1]}", f"{flat[b][0]}:{flat[b][1]}"]
                for a, b in pairs
            ],
            "n_interaction_pairs": len(pairs),
            "n_candidate_pairs": len(candidates),
            "class_counts": [int((y == 0).sum()), int((y == 1).sum())],
            "note": (
                "true interaction share is set by construction; use it to test "
                "whether an admission criterion recovers known structure. "
                "n_interaction_pairs is deliberately small -- summing many "
                "random-sign products cancels the signal toward Gaussian noise."
            ),
        }
        return blocks, y, truth


def _unit(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    return (v - float(np.mean(v))) / sd if sd > 0 else v - float(np.mean(v))
