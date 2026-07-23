# ctc-discrete-monodromy

A high-performance Rust framework for discrete operator diagnostics, symbolic metric export, and spectral stability analysis on axisymmetric background profiles.

Open-source companion to the SoftwareX **Original Software Publication** (OSP) for **ctc-discrete-monodromy** v0.1.0.

**What is reusable beyond the demo geometry?** The core product is a typed circulant-kernel / Picard harness (`DiscreteMonodromyOperator`, modular `KernelKind`, homogeneous and inhomogeneous maps `Φ = KΦ + S`, spectral-radius / Lipschitz oracles, RK4 geodesic checks). The Morris–Thorne frame-dragging profile is the reference application, not a continuum field claim.

**Naming note:** the historical `ctc-` package prefix is retained for repository continuity and does **not** assert closed-timelike-curve physics.

| Item | Value |
|------|--------|
| Crate / binary | `ctc-research-framework` / `ctc-engine` |
| Version | 0.1.0 (git tag `v0.1.0`) |
| License | MIT (`LICENSE` / `LICENSE.txt`) |
| Languages | Rust 2021 + optional Python 3 (SymPy, NumPy, SciPy, Matplotlib) |
| OS | Windows 10+, Linux, macOS |
| Repository | https://github.com/mauramyrez/ctc-discrete-monodromy |
| Support | 1912574a@umich.mx |

## Intended users (SoftwareX scope)

- Numerical analysts needing a forkable regression oracle for row-normalized dense periodic kernels
- Research software engineers who want dimension-safe Picard / spectral gates in CI
- Computational-relativity practitioners sampling explicit axisymmetric profiles
- Instructors regenerating optional illustration PDFs without coupling Python to the Rust acceptance gate

**Scope:** discrete operator diagnostics on explicit metric profiles—not continuum field theorems. Certified numerical claims refer to the row-normalized kernel \(K\) on \(\mathbb{R}^N\) and residuals of \(\Phi = K\Phi + S\).

## Quick start (acceptance gate)

```bash
git clone https://github.com/mauramyrez/ctc-discrete-monodromy.git
cd ctc-discrete-monodromy

# optional on Windows if linker path length fails:
# set CARGO_TARGET_DIR=C:\ctc-target

cargo build --release
cargo run --release --bin ctc-engine
cargo test
```

Expected: **17/17** unit tests passed.

## Capabilities

| Capability | What the crate provides |
|------------|-------------------------|
| Metric profiles | Explicit \(\{b(r),\Phi(r),\omega(r)\}\), flare-out, equatorial ergoregion / ergosurface locus |
| Discrete monodromy \(\mathcal{M}_N\) | Row-sum renormalized kernel \(K\), spectral-radius / grid-oscillation diagnostics |
| Inhomogeneous workload | Forcing vector \(S\) with Picard map \(\Phi\mapsto K\Phi+S\) (nontrivial response) |
| Modular kernels | `KernelKind::{YukawaProxy, ExponentialDecay}` comparative spectral baselines |
| Geodesics | RK4 co-rotating equatorial integrator regression |
| Optional SymPy export | Static equatorial \(G_{\mu\nu}\) / NEC–WEC documentation assets |
| Optional figures | Matplotlib scripts that regenerate SoftwareX illustration PDFs |

### Discrete operator definition

\[
K_{ij}
=
\widetilde{K}_{ij}\,
\frac{\mathrm{e}^{-m_*\Delta\tau}}{\sum_k \widetilde{K}_{ik}}
\]

Row-sum renormalization is an explicit benchmarking step and forces \(\rho(K)<1\) by construction. Homogeneous runs (\(S=0\)) converge to \(\Phi^*=0\); nonzero \(S\) yields a nontrivial discrete response. Tests enforce `field.len() == n_nodes` (no silent truncation).

## Repository layout

The GitHub repository root **is** the Cargo package root:

```
.
├── Cargo.toml                 # ctc-research-framework + [[bin]] ctc-engine
├── LICENSE / LICENSE.txt      # MIT (SoftwareX-required license file)
├── README.md
├── requirements.txt           # optional Python pins
├── docs/
│   ├── paper.tex              # SoftwareX OSP manuscript (submit this)
│   ├── softwarex-osp-template.tex
│   ├── README.md              # manuscript / template guidance
│   └── figures/               # optional illustration PDFs
├── scripts/                   # optional SymPy + Matplotlib helpers
└── src/
    ├── lib.rs
    ├── main.rs                # ctc-engine CLI
    ├── algebra/
    ├── dynamics/
    │   ├── monodromy.rs       # core discrete-operator engine
    │   ├── geodesic.rs
    │   └── bvp.rs
    ├── energy/
    └── tests/monodromy_test.rs
```

## Build options

Optional MPFR feature (needs system GMP/MPFR):

```bash
cargo build --release --features high-prec
```

### Optional Python layer (symbolic export + figures)

Not required to compile or test the Rust engine. Useful for regenerating documentation tensors and SoftwareX illustration figures:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
python scripts/derive_tensors.py
python scripts/verify_energy_conditions.py
python scripts/plot_spacetime_diagrams.py
```

Reference profile (matches `ExplicitProfileParams::prd_reference()`):

\[
(r_0,\gamma,\alpha,\omega_0,\beta)=(1.0,\ 0.5,\ 0.1,\ 1.2,\ 2.0)
\]

## Regression suite (17/17)

```bash
cargo test
# verbose monodromy output:
cargo test -- --nocapture monodromy
```

Coverage:

1. Metric / flare-out / ergosurface diagnostics  
2. Energy-scaffold API checks  
3. Monodromy bounds \(\rho(K)<1\) and oscillation Lipschitz \(L<1\)  
4. Homogeneous Picard residuals \(<10^{-8}\) for \(N\in\{32,64,128\}\)  
5. Inhomogeneous forcing \(\Phi=K\Phi+S\) with nontrivial \(\|\Phi^*\|_2\)  
6. Modular kernel spectral comparison  
7. RK4 co-rotating geodesic integration  

## Minimal API example

```rust
use ctc_research_framework::dynamics::monodromy::{
    default_forcing_vector, default_monodromy_operator,
    find_fixed_point, find_fixed_point_inhomogeneous, PeriodicField,
};

let m = default_monodromy_operator(); // N=128, m^2=1, nu=1
assert!(m.is_contraction());

let init = PeriodicField::new((0..m.n_nodes).map(|i| 1.0 + 0.1 * (i as f64)).collect());
let fp0 = find_fixed_point(&init, &m)?;
assert!(fp0.field.l2_norm() < 1e-6);

let source = default_forcing_vector(m.n_nodes);
let zero = PeriodicField::new(vec![0.0; m.n_nodes]);
let fp = find_fixed_point_inhomogeneous(&zero, &m, &source)?;
assert!(fp.field.l2_norm() > 1e-3);
assert!(fp.residual < 1e-8);
```

## Citation / manuscript

- Manuscript source: [`docs/paper.tex`](docs/paper.tex) (Elsevier `elsarticle` OSP).  
- Do **not** submit the software-update template; this is a first Original Software Publication.  
- See [`docs/README.md`](docs/README.md) for template guidance.

## License

MIT. Copyright (c) 2026 Mauricio M. Paniagua.  
Support: 1912574a@umich.mx
