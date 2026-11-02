"""Multi-register feature map for radiomics + transcriptomics fusion.

Registers (authors' packing): [Shape, Texture, Transcriptomics]

Design note (authors, 2026-08):
    "Hadamard init vs Angle Encoding" is a false dichotomy. H^{on}+RZ(x) *is*
    angle encoding, just in the X basis (this is literally what ZZFeatureMap
    does: H then P(2x)). The real axes of choice are:

      axis 1 - basis:  RY from |0>  (Y/Z-plane, amplitude-visible)
                       H + RZ       (phase, invisible to Z-measurement alone)
      axis 2 - entanglement topology: block-local vs cross-register

    Consequence: H + RZ with NO entangler and a Z-basis readout produces a
    state whose measurement statistics are *identical for every input*. The
    phases only become observable after an entangler converts them into
    populations. See scripts/encoding_probe.py for the empirical proof.

Encoders provided
-----------------
    ry        : RY(x) on |0>            - dense, no entangler needed
    h_rz      : H then RZ(x)            - phase-only (degenerate alone)
    zz        : H, RZ(x), CX-RZ-CX      - ZZFeatureMap-equivalent
    ry_zz     : RY(x) then ZZ entangler - hybrid, recommended v1 default

Coupling
--------
    none      : registers stay independent (fusion happens only in the ansatz)
    ring      : adjacent register pairs, wrapping (shape->texture->omics->shape)
    full      : all register pairs coupled (expensive, keep for ablation)

Coupler gate matters as much as topology:
    crz : diagonal -> commutes with Z, can NEVER move a Z-marginal. Cross-register
          information lives in phases and stays invisible to the readout.
    cry : non-diagonal -> actually transports information between registers.
    Probed empirically in scripts/encoding_probe.py (D3); crz scored 0.00000.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_REGISTERS = ("shape", "texture", "transcriptomics")


@dataclass
class RegisterSpec:
    """One feature block -> one contiguous qubit register."""

    name: str
    n_qubits: int
    # per-register angle scale; transcriptomics is usually noisier so a
    # smaller scale keeps it from dominating the state.
    angle_scale: float = np.pi / 2

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError(f"register {self.name!r}: n_qubits must be >= 1")


@dataclass
class MultiRegisterConfig:
    registers: list[RegisterSpec] = field(
        default_factory=lambda: [
            RegisterSpec("shape", 3),
            RegisterSpec("texture", 4),
            RegisterSpec("transcriptomics", 4, angle_scale=np.pi / 4),
        ]
    )
    encoder: str = "ry_zz"       # ry | h_rz | zz | ry_zz
    coupling: str = "ring"       # none | ring | full
    coupler: str = "cry"         # crz (phase-only) | cry (transports info)
    reps: int = 1

    @property
    def n_qubits(self) -> int:
        return sum(r.n_qubits for r in self.registers)

    def slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for r in self.registers:
            out[r.name] = slice(start, start + r.n_qubits)
            start += r.n_qubits
        return out


# --------------------------------------------------------------------------- #
# circuit construction
# --------------------------------------------------------------------------- #
def build_feature_map(cfg: MultiRegisterConfig):
    """Return a parameterised QuantumCircuit with one Parameter per feature."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n = cfg.n_qubits
    qc = QuantumCircuit(n, name=f"mrfm_{cfg.encoder}_{cfg.coupling}")
    x = ParameterVector("x", n)
    spans = cfg.slices()

    for _ in range(cfg.reps):
        for reg in cfg.registers:
            sl = spans[reg.name]
            qubits = list(range(sl.start, sl.stop))
            _encode_block(qc, qubits, x, sl, reg.angle_scale, cfg.encoder)
        _couple_registers(qc, cfg, spans, x)
        qc.barrier()
    return qc, x


def _encode_block(qc, qubits, x, sl, scale, encoder):
    idx = list(range(sl.start, sl.stop))

    if encoder == "ry":
        for q, i in zip(qubits, idx):
            qc.ry(scale * x[i], q)
        return

    if encoder in ("h_rz", "zz"):
        for q, i in zip(qubits, idx):
            qc.h(q)
            qc.rz(2.0 * scale * x[i], q)
    elif encoder == "ry_zz":
        for q, i in zip(qubits, idx):
            qc.ry(scale * x[i], q)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")

    # intra-register ZZ entangler (linear chain, not all-to-all: with <60
    # samples an all-to-all map has far more expressivity than we can fit)
    if encoder in ("zz", "ry_zz"):
        for (a, ia), (b, ib) in zip(zip(qubits, idx), zip(qubits[1:], idx[1:])):
            qc.cx(a, b)
            qc.rz(2.0 * scale * (np.pi - x[ia]) * (np.pi - x[ib]) / np.pi, b)
            qc.cx(a, b)


def _couple_registers(qc, cfg, spans, x):
    """Cross-register interaction - this is the multi-omics fusion hypothesis."""
    if cfg.coupling == "none":
        return
    names = [r.name for r in cfg.registers]
    if cfg.coupling == "ring":
        # wrap around, otherwise the last register can never reach the first
        pairs = list(zip(names, names[1:] + names[:1]))
        if len(names) == 2:
            pairs = pairs[:1]
    elif cfg.coupling == "full":
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    else:
        raise ValueError(f"unknown coupling {cfg.coupling!r}")

    for a, b in pairs:
        qa, qb = spans[a].start, spans[b].start   # head qubit of each register
        theta = 0.5 * (x[qa] + x[qb])
        if cfg.coupler == "crz":
            qc.crz(theta, qa, qb)
        elif cfg.coupler == "cry":
            qc.cry(theta, qa, qb)
        else:
            raise ValueError(f"unknown coupler {cfg.coupler!r}")


def circuit_stats(cfg: MultiRegisterConfig) -> dict:
    qc, _ = build_feature_map(cfg)
    d = qc.decompose()
    return {
        "encoder": cfg.encoder,
        "coupling": cfg.coupling,
        "coupler": cfg.coupler,
        "n_qubits": qc.num_qubits,
        "n_parameters": qc.num_parameters,
        "depth": qc.depth(),
        "depth_decomposed": d.depth(),
        "two_qubit_gates": sum(1 for i in qc.data if i.operation.num_qubits == 2),
    }


__all__ = [
    "RegisterSpec",
    "MultiRegisterConfig",
    "build_feature_map",
    "circuit_stats",
    "DEFAULT_REGISTERS",
]
