# Supplementary numerical material (Classical and Quantum Gravity)

Reproducible companion code and data for the CQG manuscript

**Discrete Spacelike Monodromy Proxy and Infrared Spectral Transition on Axisymmetric Frame-Dragging Backgrounds**

This repository is **not** a standalone software product. It archives the discrete monodromy diagnostics, symbolic tensor expansions, and spectral tables that support the theoretical claims of the paper—chiefly criterion **(C3)**: the mass-gap / massless spectral rupture of the unnormalized proxy \(\widetilde{K}\) on a spacelike azimuthal loop (Sec.&nbsp;IV).

| Item | Value |
|------|--------|
| Manuscript | [`docs/paper-cqg.tex`](docs/paper-cqg.tex) / [`docs/paper-cqg.pdf`](docs/paper-cqg.pdf) (IOP `iopjournal` + `iopart-num`; CQG) |
| Spectral tables (Path-3) | [`docs/path3_continuum_report.md`](docs/path3_continuum_report.md) |
| Crate / binary | `ctc-research-framework` / `ctc-engine` |
| Version | 0.1.0 (git tag `v0.1.0`) |
| License | MIT (`LICENSE` / `LICENSE.txt`) |
| Languages | Rust 2021 + optional Python 3 (SymPy, NumPy, SciPy, Matplotlib) |
| Repository | https://github.com/mauramyrez/ctc-discrete-monodromy |
| Contact | 1912574a@umich.mx |

**Naming note.** The historical `ctc-` package prefix is retained for repository continuity and does **not** by itself assert continuum closed-timelike-curve field theorems. All certified numerical claims refer to the discrete operators \(K\) and \(\widetilde{K}\) constructed in the manuscript.

## Scope (aligned with the paper)

- Evaluates a **discrete monodromy proxy** on a fixed-\((r,\theta)\) azimuthal loop: spectral radii, contraction diagnostics, and Picard fixed points.
- Does **not** solve a fully dynamical Klein–Gordon field on the curved background, evolve the Einstein equations, or reconstruct a continuum Green function.
- Row-sum renormalization of \(K\) is an algebraic control (\(\rho(K)<1\) by construction). The physical infrared diagnostic is the **unnormalized** proxy \(\widetilde{K}\) under \(m^{2}\to 0\) at \(\nu=0\).
- Semiclassical Hadamard / EFT bounds in the manuscript (criterion C4) are an analytic outlook and are **not** numerical outputs of this code.

Reference profile (matches `ExplicitProfileParams::prd_reference()` and Eq.&nbsp;(3) of the paper):

\[
(r_0,\gamma,\alpha,\omega_0,\beta)=(1.0,\ 0.5,\ 0.1,\ 1.2,\ 2.0),\qquad r_{\mathrm{loop}}=1.5\,r_0.
\]

## Data and code access

The discrete-monodromy source code and full spectral tables accompanying this work are available at https://github.com/mauramyrez/ctc-discrete-monodromy; the Path-3 continuum campaign report is archived as `docs/path3_continuum_report.md` in that repository.

Regenerate the Path-3 sweep (unnormalized \(\widetilde{K}\), \(\nu=0\), \(N\le 1024\)):

```bash
cargo test --release path3_extended -- --nocapture
```

## Reproduce the numerical gates

```bash
git clone https://github.com/mauramyrez/ctc-discrete-monodromy.git
cd ctc-discrete-monodromy

# optional on Windows if linker path length fails:
# set CARGO_TARGET_DIR=C:\ctc-target

cargo build --release
cargo run --release --bin ctc-engine
cargo test
```

Expected: **17/17** unit tests passed (geometry / energy-scaffold checks, renormalized contraction, inhomogeneous Picard residuals, modular kernels, RK4 geodesics).

Optional high-precision build (system GMP/MPFR required):

```bash
cargo build --release --features high-prec
```

## Compile the manuscript

From `docs/` (requires MiKTeX / TeX Live; uses local `iopjournal.cls`):

```bash
pdflatex paper-cqg
bibtex paper-cqg
pdflatex paper-cqg
pdflatex paper-cqg
```

See [`docs/README.md`](docs/README.md) for the file map (appendices, figures, Path-3 report).

## What the code implements

| Capability | Role relative to the paper |
|------------|----------------------------|
| Explicit profiles \(\{b(r),\Phi(r),\omega(r)\}\) | Geometry / ergoregion locus (C1) |
| Symbolic equatorial \(G_{\mu\nu}\) / NEC–WEC export | Static background constraints (C2) |
| Row-sum renormalized monodromy \(K\) | Algebraic Banach–Picard gate (C3 control) |
| Unnormalized proxy \(\widetilde{K}\) (Yukawa vs exponential) | Mass-gap threshold / IR rupture (C3 core) |
| RK4 co-rotating geodesics | Supporting geometric diagnostics |
| Optional Matplotlib figures | Illustration PDFs under `docs/figures/` |

Discrete operator (row-sum renormalized control gate):

\[
K_{ij}
=
\widetilde{K}_{ij}\,
\frac{\mathrm{e}^{-m_*\Delta\tau}}{\sum_k \widetilde{K}_{ik}}.
\]

## Repository layout

```
.
├── Cargo.toml                 # ctc-research-framework + [[bin]] ctc-engine
├── LICENSE / LICENSE.txt
├── README.md                  # this file (CQG supplementary material)
├── requirements.txt           # optional Python pins
├── docs/
│   ├── paper-cqg.tex / .pdf   # CQG manuscript (iopjournal + iopart-num)
│   ├── iopjournal.cls         # IOP submission class
│   ├── iopart-num.bst
│   ├── references.bib
│   ├── path3_continuum_report.md
│   ├── figures/               # lightcones, geodesic loop, …
│   └── README.md
├── scripts/                   # optional SymPy + Matplotlib helpers
└── src/
    ├── lib.rs
    ├── main.rs
    ├── dynamics/monodromy.rs  # discrete-operator engine
    └── tests/monodromy_test.rs
```


### Optional Python layer

Not required to compile or test the Rust engine. Useful for regenerating symbolic tensors and figure PDFs:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
python scripts/derive_tensors.py
python scripts/verify_energy_conditions.py
python scripts/plot_spacetime_diagrams.py
```

## Citation

Please cite the Classical and Quantum Gravity manuscript (`docs/paper-cqg.tex`). For the numerical companion, point readers to this repository and to `docs/path3_continuum_report.md`.

## License

MIT. Copyright (c) 2026 Mauricio M. Paniagua.  
Contact: 1912574a@umich.mx
