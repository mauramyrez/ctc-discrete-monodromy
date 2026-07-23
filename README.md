# Discrete Sobolev Monodromy Contraction and Novikov Consistency Verification on Axisymmetric Frame-Dragging Wormhole Backgrounds

Computational companion to the IOP Classical and Quantum Gravity manuscript of the same title: symbolic GR export (SymPy), typed discrete monodromy integration (Rust), and figure generation (Matplotlib).

## Conventions

| Item | Choice |
|------|--------|
| Metric signature | \((-+++\) |
| Units | \(c = G = 1\) (geometric) |
| Summation | Einstein convention |
| Journal target | Classical and Quantum Gravity (IOP; `iopjournal.cls`) |

## Computational Status & Architecture

| Criterion | Status | What the repository provides |
|-----------|--------|------------------------------|
| **(C1)** | Implemented | Explicit profiles \(\{b(r),\Phi(r),\omega(r)\}\), flare-out, CTC locus bisection |
| **(C2)** | Symbolic export | Equatorial \(G_{\mu\nu}\) / NEC–WEC via SymPy as **static** background assets (not dynamically assembled in Rust) |
| **(C3)** | **Core verified result** | Discrete Yukawa-proxy monodromy \(\mathcal{M}_N\): row-sum renormalized kernel \(K\), discrete \(H^1\)-proxy contraction, Banach–Picard fixed point in Rust |
| **(C4)** | Analytic outlook only | Curvature-cutoff Hadamard bound — **not** numerically computed by this engine |

### Discrete operator definition (C3)

The continuous Yukawa form with mode-augmented mass \(m_*\) **motivates** the kernel; all certified claims refer to the discrete operator on \(\mathbb{R}^N\):

\[
K_{ij}
=
\widetilde{K}_{ij}\,
\frac{\mathrm{e}^{-m_*\Delta\tau}}{\sum_k \widetilde{K}_{ik}}
\]

**Row-sum renormalization by \(\mathrm{e}^{-m_*\Delta\tau}\) is an explicit step of the discrete definition of \(\mathcal{M}_N\)**, not an emergent continuum property. Tests enforce `field.len() == n_nodes` (no silent truncation).

Energy-condition code paths are an **API scaffold** for quadratic-form sanity checks only; manuscript NEC/WEC contractions come from SymPy.

## Layout

```
CTC-Research-Framework/
├── docs/           # paper.tex, references.bib, figures/, appendices
├── scripts/        # SymPy derivations & Matplotlib diagrams
├── src/            # Rust: algebra / dynamics (M_N) / energy (scaffold)
├── Cargo.toml
└── requirements.txt
```

## Quick start

### Python (symbolic + figures)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
# source .venv/bin/activate
pip install -r requirements.txt
python scripts/derive_tensors.py
python scripts/verify_energy_conditions.py
python scripts/plot_spacetime_diagrams.py
```

Reference profile (matches `ExplicitProfileParams::prd_reference()`):

\[
(r_0,\gamma,\alpha,\omega_0,\beta)=(1.0,\ 0.5,\ 0.1,\ 1.2,\ 2.0)
\]

### Rust (discrete monodromy)

Requires a Rust toolchain (`rustup`). On Windows with long OneDrive paths, prefer a short target directory:

```bash
# optional on Windows if linker path length fails:
# set CARGO_TARGET_DIR=C:\ctc-target

cargo build --release
cargo run --release --bin ctc-engine
cargo test
```

Expected: **17/17** unit tests passed (geometry, energy scaffold, \(\mathcal{M}_N\) contraction, Picard for \(N\in\{32,64,128\}\), RK4 geodesics).

Optional MPFR via `rug` (needs system GMP/MPFR):

```bash
cargo build --release --features high-prec
```

## Workflow

1. **(C2)** — Export tensors with `scripts/derive_tensors.py` / `verify_energy_conditions.py`.
2. **(C1)/(C3)** — Profiles and \(\mathcal{M}_N\) live under `src/algebra` and `src/dynamics/novikov.rs`; run `cargo test`.
3. **Manuscript** — Compile `docs/paper.tex` with latexmk / pdflatex + bibtex (figures from `docs/figures/`).

## License

MIT (research code). Manuscript copyright remains with the sole author, Mauricio Miguel Paniagua Ramirez.
