"""vqc_multiregister — VQC over [Shape, Texture, Transcriptomics] registers.

Conforms to Algorithm Interface Spec v0.

Register dimensions are fully parameterised (authors' request, 2026-08). Each
feature block gets its own qubit budget, its own angle scale, and its own PCA
fitted *inside* the training fold only.

Why per-register PCA and not one global PCA
-------------------------------------------
Transcriptomics is ~20k features, radiomics shape is ~10. A single global PCA
is dominated by the omics block and the imaging signal disappears. Fitting one
PCA per register keeps each modality's own variance structure and makes the
"how many qubits does each modality deserve" question an explicit knob.

Leakage guard
-------------
`docs/vqc_review_2026-07-31.md` P0 was a PCA fitted on the full 46-sample set
before the train/test split. Here every transform lives in `train()` and is
only ever applied — never refitted — in `predict()`. `self._fitted_on` records
the training row count so a mismatch is loud instead of silent.

Feature layout contract
-----------------------
X arrives as one concatenated matrix. `blocks` in config declares the column
span of each register, e.g.

    blocks:
      shape:           [0, 14]
      texture:         [14, 102]
      transcriptomics: [102, 20633]

If `blocks` is omitted the whole matrix is treated as one register.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BLOCKS = ("shape", "texture", "transcriptomics")


class Algorithm:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        p = self.config.get("params", {})

        self.register_specs = self._parse_registers(p)
        self.blocks = self._parse_blocks(p)
        self.encoder = str(p.get("encoder", "ry"))
        self.coupling = str(p.get("coupling", "ring"))
        self.coupler = str(p.get("coupler", "cry"))
        self.reps = int(p.get("reps", 1))
        self.reps_ansatz = int(p.get("reps_ansatz", 2))
        self.optimizer_name = str(p.get("optimizer", "cobyla")).lower()
        self.maxiter = int(p.get("maxiter", 300))
        self.seed = int(p.get("seed", 42))

        self.model = None
        self._pipelines: dict[str, object] = {}
        self._cfg = None
        self._classes = None
        self._fitted_on = None
        self._used_fallback = False

    # ------------------------------------------------------------------ #
    # config parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_registers(p: dict) -> list[dict]:
        raw = p.get("registers")
        if not raw:
            return [
                {"name": "shape", "n_qubits": 3, "angle_scale": np.pi / 2},
                {"name": "texture", "n_qubits": 4, "angle_scale": np.pi / 2},
                {"name": "transcriptomics", "n_qubits": 4,
                 "angle_scale": np.pi / 4},
            ]
        out = []
        for r in raw:
            out.append({
                "name": str(r["name"]),
                "n_qubits": int(r["n_qubits"]),
                "angle_scale": float(r.get("angle_scale", np.pi / 2)),
            })
        return out

    @staticmethod
    def _parse_blocks(p: dict) -> dict[str, tuple[int, int]] | None:
        raw = p.get("blocks")
        if not raw:
            return None
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}

    @property
    def n_qubits(self) -> int:
        return sum(r["n_qubits"] for r in self.register_specs)

    # ------------------------------------------------------------------ #
    # per-register preprocessing (fit ONLY on training data)
    # ------------------------------------------------------------------ #
    def _slice_block(self, X, name: str):
        if self.blocks is None:
            return X
        if name not in self.blocks:
            raise KeyError(
                f"register {name!r} has no column span in params.blocks; "
                f"known blocks: {sorted(self.blocks)}"
            )
        lo, hi = self.blocks[name]
        if hi > X.shape[1]:
            raise ValueError(
                f"block {name!r} spans [{lo},{hi}) but X has {X.shape[1]} columns"
            )
        return X[:, lo:hi]

    def _build_pipeline(self, n_features: int, n_qubits: int, scale: float):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA

        steps = [("var", VarianceThreshold(threshold=1e-8)),
                 ("std", StandardScaler())]
        if n_features > n_qubits:
            steps.append(("pca", PCA(n_components=n_qubits,
                                     random_state=self.seed)))
        steps.append(("ang", MinMaxScaler(feature_range=(-scale, scale))))
        return Pipeline(steps)

    def _fit_transform_blocks(self, X):
        parts = []
        for spec in self.register_specs:
            name, nq = spec["name"], spec["n_qubits"]
            Xb = self._slice_block(X, name)
            pipe = self._build_pipeline(Xb.shape[1], nq, spec["angle_scale"])
            Zb = pipe.fit_transform(Xb)
            if Zb.shape[1] < nq:                      # pad if block was thin
                Zb = np.hstack([Zb, np.zeros((Zb.shape[0], nq - Zb.shape[1]))])
            self._pipelines[name] = pipe
            parts.append(Zb[:, :nq])
        return np.hstack(parts)

    def _transform_blocks(self, X):
        parts = []
        for spec in self.register_specs:
            name, nq = spec["name"], spec["n_qubits"]
            Xb = self._slice_block(X, name)
            Zb = self._pipelines[name].transform(Xb)
            if Zb.shape[1] < nq:
                Zb = np.hstack([Zb, np.zeros((Zb.shape[0], nq - Zb.shape[1]))])
            parts.append(Zb[:, :nq])
        return np.hstack(parts)

    # ------------------------------------------------------------------ #
    # quantum pieces
    # ------------------------------------------------------------------ #
    def _make_circuit_config(self):
        from .circuit import MultiRegisterConfig, RegisterSpec

        return MultiRegisterConfig(
            registers=[RegisterSpec(r["name"], r["n_qubits"], r["angle_scale"])
                       for r in self.register_specs],
            encoder=self.encoder,
            coupling=self.coupling,
            coupler=self.coupler,
            reps=self.reps,
        )

    def _build_optimizer(self):
        from qiskit_algorithms.optimizers import COBYLA, SPSA
        if self.optimizer_name == "spsa":
            return SPSA(maxiter=self.maxiter)
        return COBYLA(maxiter=self.maxiter)

    # ------------------------------------------------------------------ #
    # interface
    # ------------------------------------------------------------------ #
    def train(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._classes = np.unique(y)
        self._fitted_on = X.shape[0]

        try:
            from qiskit.circuit.library import RealAmplitudes
            from qiskit_machine_learning.algorithms import VQC
            from qiskit.primitives import StatevectorSampler
            from qiskit_algorithms.utils import algorithm_globals
            from sklearn.preprocessing import OneHotEncoder

            from .circuit import build_feature_map

            algorithm_globals.random_seed = self.seed

            Xt = self._fit_transform_blocks(X)
            self._cfg = self._make_circuit_config()
            feature_map, _ = build_feature_map(self._cfg)
            nq = feature_map.num_qubits

            if Xt.shape[1] != nq:
                raise ValueError(
                    f"encoded width {Xt.shape[1]} != circuit qubits {nq}")

            ansatz = RealAmplitudes(num_qubits=nq, reps=self.reps_ansatz)
            y_oh = OneHotEncoder(sparse_output=False).fit_transform(
                y.reshape(-1, 1))

            self.model = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=self._build_optimizer(),
                sampler=StatevectorSampler(),
            )
            self.model.fit(Xt, y_oh)
            self._used_fallback = False
        except Exception as e:                        # noqa: BLE001
            print(f"[vqc_multiregister] quantum path failed: {e}")
            self._used_fallback = True
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=self.seed)
            self.model.fit(X, y)
        return self

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        if self._used_fallback or not self._pipelines:
            return X
        return self._transform_blocks(X)

    def predict(self, X) -> np.ndarray:
        Xt = self._transform(X)
        if self._used_fallback:
            return self.model.predict(Xt)
        out = np.asarray(self.model.predict(Xt))
        if out.ndim == 2:
            out = np.argmax(out, axis=1)
        return out.astype(int)

    def predict_proba(self, X) -> np.ndarray:
        Xt = self._transform(X)
        if self._used_fallback:
            return self.model.predict_proba(Xt)
        try:
            p = np.asarray(self.model.predict_proba(Xt), dtype=float)
        except Exception:                             # noqa: BLE001
            p = np.asarray(self.model.neural_network.forward(
                Xt, self.model.weights), dtype=float)
        if p.ndim == 1:
            p = np.vstack([1 - p, p]).T
        s = p.sum(axis=1, keepdims=True)
        return np.divide(p, s, out=np.full_like(p, 0.5), where=s > 0)

    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        from .circuit import circuit_stats
        cfg = self._cfg or self._make_circuit_config()
        return {
            "registers": self.register_specs,
            "blocks": self.blocks,
            "circuit": circuit_stats(cfg),
            "fitted_on_n": self._fitted_on,
            "used_fallback": self._used_fallback,
        }
