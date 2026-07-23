# ctc-discrete-monodromy

A high-performance Rust framework for discrete operator diagnostics, symbolic metric export, and spectral stability analysis on axisymmetric background profiles.

This repository is the computational companion to the SoftwareX Original Software Publication for **ctc-discrete-monodromy** (v0.1.0). It provides typed circulant-kernel assembly, spectral-radius / Lipschitz regression oracles, homogeneous and inhomogeneous Picard solvers, RK4 geodesic checks, and SymPy export of equatorial curvature assets.

**Scope:** discrete operator diagnostics on explicit metric profiles—not continuum field theorems. Certified numerical claims refer to the row-normalized kernel \(K\) on \(\mathbb{R}^N\) and residuals of the map \(\Phi = K\Phi + S\).

| Item | Value |
|------|--------|
| Package | `ctc-research-framework` 0.1.0 |
| Binary | `ctc-engine` |
| License | MIT |
| Languages | Rust 2021 + Python 3 (SymPy / NumPy / SciPy / Matplotlib) |
| OS | Windows 10+, Linux, macOS |
| Repository | https://github.com/mauramyrez/ctc-discrete-monodromy |

## Capabilities

| Capability | What the crate provides |
|------------|-------------------------|
| Metric profiles | Explicit \(\{b(r),\Phi(r),\omega(r)\}\), flare-out, equatorial ergoregion / ergosurface locus |
| Symbolic export | Equatorial \(G_{\mu\nu}\) / NEC–WEC via SymPy as **static** background assets |
| Discrete monodromy \(\mathcal{M}_N\) | Row-sum renormalized kernel \(K\), spectral-radius / grid-oscillation diagnostics |
| Inhomogeneous workload | Forcing vector \(S\) with Picard map \(\Phi\mapsto K\Phi+S\) (nontrivial response) |
| Modular kernels | `KernelKind::{YukawaProxy, ExponentialDecay}` comparative spectral baselines |
| Geodesics | RK4 co-rotating equatorial integrator regression |

### Discrete operator definition

The continuous Yukawa form with mode-augmented mass \(m_*\) **motivates** the kernel; all certified claims refer to the discrete operator on \(\mathbb{R}^N\):

\[
K_{ij}
=
\widetilde{K}_{ij}\,
\frac{\mathrm{e}^{-m_*\Delta\tau}}{\sum_k \widetilde{K}_{ik}}
\]

Row-sum renormalization by \(\mathrm{e}^{-m_*\Delta\tau}\) is an explicit benchmarking step. It forces \(\rho(K)<1\) by construction. The homogeneous map (\(S=0\)) converges to \(\Phi^*=0\); a nonzero forcing \(S\) yields a nontrivial discrete response \(\Phi^*=(I-K)^{-1}S\). Tests enforce `field.len() == n_nodes` (no silent truncation).

## Cargo workspace layout

This crate is a single-package Cargo project (not a multi-crate workspace). Layout:

```
CTC-Research-Framework/          # crate root (ctc-research-framework)
├── Cargo.toml                   # package + [[bin]] ctc-engine
├── LICENSE                      # MIT
├── README.md
├── requirements.txt             # Python pins for SymPy export / figures
├── docs/                        # SoftwareX manuscript (paper.tex) + figures
├── scripts/                     # SymPy derivations & Matplotlib diagrams
└── src/
    ├── lib.rs                   # algebra / dynamics / energy
    ├── main.rs                  # ctc-engine CLI
    ├── algebra/                 # metric profiles, invariants, Christoffel FD
    ├── dynamics/
    │   ├── monodromy.rs         # DiscreteMonodromyOperator, Picard, kernels
    │   ├── geodesic.rs          # RK4 integrator
    │   └── bvp.rs               # smoke-test re-exports
    ├── energy/                  # quadratic-form API scaffold
    └── tests/monodromy_test.rs  # regression suite entry
```

## Build instructions

Requires a Rust toolchain via [`rustup`](https://rustup.rs/) (stable) and, for the Python layer, Python ≥ 3.10.

### Rust (discrete operator engine)

On Windows with long OneDrive paths, prefer a short Cargo target directory:

```bash
# optional on Windows if linker path length fails:
# set CARGO_TARGET_DIR=C:\ctc-target

cd CTC-Research-Framework
cargo build --release
cargo run --release --bin ctc-engine
```

Optional MPFR high-precision feature (needs system GMP/MPFR):

```bash
cargo build --release --features high-prec
```

### Python (symbolic metric export + figures)

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

## Running the 17/17 regression suite

From the crate root:

```bash
cargo test
```

Expected: **17/17** unit tests passed, covering

1. Metric / flare-out / ergosurface diagnostics
2. Energy-scaffold API checks
3. Monodromy bounds \(\rho(K)<1\) and oscillation Lipschitz \(L<1\)
4. Homogeneous Picard residuals \(<10^{-8}\) for \(N\in\{32,64,128\}\)
5. Inhomogeneous forcing \(\Phi=K\Phi+S\) with nontrivial \(\|\Phi^*\|_2\)
6. Modular kernel spectral comparison (`YukawaProxy` vs `ExponentialDecay`)
7. RK4 co-rotating geodesic integration

Verbose monodromy output:

```bash
cargo test -- --nocapture monodromy
```

## Minimal API example

```rust
use ctc_research_framework::dynamics::monodromy::{
    default_forcing_vector, default_monodromy_operator,
    find_fixed_point, find_fixed_point_inhomogeneous, PeriodicField,
};

let m = default_monodromy_operator(); // N=128, m^2=1, nu=1
assert!(m.is_contraction());

// Homogeneous baseline → Φ* ≈ 0
let init = PeriodicField::new((0..m.n_nodes).map(|i| 1.0 + 0.1 * (i as f64)).collect());
let fp0 = find_fixed_point(&init, &m)?;
assert!(fp0.field.l2_norm() < 1e-6);

// Inhomogeneous workload → nontrivial response
let source = default_forcing_vector(m.n_nodes);
let zero = PeriodicField::new(vec![0.0; m.n_nodes]);
let fp = find_fixed_point_inhomogeneous(&zero, &m, &source)?;
assert!(fp.field.l2_norm() > 1e-3);
assert!(fp.residual < 1e-8);
```

## License

MIT. Manuscript copyright remains with the sole author, Mauricio Miguel Paniagua Ramirez.
Support: 1912574a@umich.mx
