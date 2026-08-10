//! Discrete operator diagnostics for circulant kernels on \(\mathbb{R}^N\).
//!
//! This module builds explicit metric-sampled kernels on azimuthal loops and exposes
//! spectral / Picard regression oracles. Continuous Yukawa structure only *motivates*
//! kernel shapes.
//!
//! Two assembly policies are supported via [`RenormPolicy`]:
//! - [`RenormPolicy::RowSumBenchmark`]: row-sum renormalization
//!   \(K_{ij}=\widetilde{K}_{ij}\,\mathrm{e}^{-m_*\Delta\tau}/\sum_k\widetilde{K}_{ik}\)
//!   (legacy SoftwareX / regression gate; contractive by construction when \(m_*>0\)).
//! - [`RenormPolicy::None`]: physical proxy kernel \(\widetilde{K}\) with **no** forced
//!   row-sum fix — used for Path-3 continuum stress tests (\(\rho(\widetilde{K}_N)\) vs \(N,m^2\)).
//!
//! Homogeneous baselines (\(S=0\)) under a contraction gate converge to \(\Phi^*=0\);
//! nonzero forcing \(S\) yields \(\Phi^*=(I-K)^{-1}S\) when \(\rho(K)<1\).

use crate::algebra::metric_numeric::{metric_at, MorrisThorneParams};

use anyhow::Result;

/// Modular discrete kernel family used to assemble \(\widetilde{K}\) before optional renormalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    /// Yukawa-proxy sample \(m_*/(4\pi\,ds)\,\mathrm{e}^{-m_* ds}\) (default regression kernel).
    YukawaProxy,
    /// Pure exponential decay \(\mathrm{e}^{-m_* ds}\) (comparative modular baseline).
    ExponentialDecay,
}

impl Default for KernelKind {
    fn default() -> Self {
        Self::YukawaProxy
    }
}

/// Row-assembly policy for the discrete monodromy kernel.
///
/// Path 3 of the research audit: [`RenormPolicy::None`] exposes the physical proxy
/// \(\widetilde{K}\) so continuum contractivity can fail; [`RenormPolicy::RowSumBenchmark`]
/// preserves the historical forced-contraction regression behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RenormPolicy {
    /// Physical / continuum-stress kernel: return \(\widetilde{K}\) without row-sum scaling.
    None,
    /// Legacy benchmarking: force every row sum to \(\mathrm{e}^{-m_*\Delta\tau}<1\).
    #[default]
    RowSumBenchmark,
}

/// Discrete field samples \(\Phi_i\) on uniform nodes \(\chi_i = 2\pi i / N\).
#[derive(Clone, Debug)]
pub struct PeriodicField {
    pub values: Vec<f64>,
}

impl PeriodicField {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn l2_norm(&self) -> f64 {
        (self.values.iter().map(|v| v * v).sum::<f64>() / self.len() as f64).sqrt()
    }

    /// Discrete grid-oscillation proxy norm on \(\mathbb{R}^N\) (index-space; not continuum \(H^1\)).
    pub fn h1_norm(&self) -> f64 {
        oscillation_norm_slice(&self.values)
    }

    /// Homogeneous residual \(\|\Phi - K\Phi\|_2\).
    pub fn fixed_point_residual(&self, kernel: &[Vec<f64>]) -> f64 {
        self.inhomogeneous_residual(kernel, &vec![0.0; self.len()])
    }

    /// Inhomogeneous residual \(\|\Phi - K\Phi - S\|_2\).
    pub fn inhomogeneous_residual(&self, kernel: &[Vec<f64>], source: &[f64]) -> f64 {
        let n = self.len();
        assert_eq!(source.len(), n, "source.len() must equal field.len()");
        let mapped = apply_kernel(kernel, &self.values);
        let mut acc = 0.0;
        for i in 0..n {
            let d = self.values[i] - mapped[i] - source[i];
            acc += d * d;
        }
        (acc / n as f64).sqrt()
    }
}

/// Discrete grid-oscillation proxy norm for a periodic sample vector.
pub fn oscillation_norm_slice(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return (v.iter().map(|x| x * x).sum::<f64>() / n.max(1) as f64).sqrt();
    }
    let l2_sq: f64 = v.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let mut grad_sq = 0.0;
    for i in 0..n {
        let diff = (v[(i + 1) % n] - v[(i + n - 1) % n]) / 2.0;
        grad_sq += diff * diff;
    }
    grad_sq /= n as f64;
    (l2_sq + grad_sq).sqrt()
}

/// Alias retained for callers that still use the historical name.
pub fn h1_norm_slice(v: &[f64]) -> f64 {
    oscillation_norm_slice(v)
}

/// Matrix–vector product for the row-sum renormalized discrete kernel \(K\).
///
/// Requires `field.len() == kernel.len()` (square \(N\times N\) acting on \(\mathbb{R}^N\)).
pub fn apply_kernel(kernel: &[Vec<f64>], field: &[f64]) -> Vec<f64> {
    let n = kernel.len();
    assert!(!kernel.is_empty(), "discrete kernel K must be non-empty");
    assert_eq!(
        field.len(),
        n,
        "dimension mismatch: field.len()={} != n_nodes={} (silent truncation forbidden)",
        field.len(),
        n
    );
    for (i, row) in kernel.iter().enumerate() {
        assert_eq!(
            row.len(),
            n,
            "kernel row {i} has length {}, expected {n}",
            row.len()
        );
    }
    let mut out = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            out[i] += kernel[i][j] * field[j];
        }
    }
    out
}

/// Discrete monodromy / circulant-kernel operator \(\mathcal{M}_N\) on \(\mathbb{R}^N\).
///
/// Builds a proxy kernel from a modular sample with mode-augmented mass \(m_*\).
/// Whether row-sum renormalization is applied is controlled by [`RenormPolicy`].
/// Certified claims are discrete-operator diagnostics only.
#[derive(Clone, Debug)]
pub struct DiscreteMonodromyOperator {
    /// Loop radius \(r > r_0\) (outside throat).
    pub r_loop: f64,
    /// Background metric parameters at `r_loop`.
    pub params: MorrisThorneParams,
    /// Bare mass squared \(m^2 > 0\) (geometric units).
    pub m_squared: f64,
    /// Azimuthal mode number \(\nu\) in \(e^{i\nu\varphi}\).
    pub azimuthal_mode: i32,
    /// Number of nodes \(N\) along the azimuthal loop.
    pub n_nodes: usize,
    /// Modular kernel family for \(\widetilde{K}\) assembly.
    pub kernel_kind: KernelKind,
    /// Row-assembly policy (`RowSumBenchmark` = legacy; `None` = physical \(\widetilde{K}\)).
    pub renorm: RenormPolicy,
}

/// Alias retained for callers that prefer the shorter paper-facing name.
pub type MonodromyOperator = DiscreteMonodromyOperator;

impl DiscreteMonodromyOperator {
    fn dphi(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.n_nodes as f64
    }

    fn g_phiphi(&self) -> f64 {
        let (g, _) = metric_at(self.r_loop, std::f64::consts::FRAC_PI_2, &self.params);
        g[3][3].max(1e-12)
    }

    /// Mode-augmented mass \(m_* = \sqrt{m^2 + \nu^2/g_{\varphi\varphi}}\).
    fn m_star(&self) -> f64 {
        let m_mode = self.azimuthal_mode as f64;
        let g_pp = self.g_phiphi();
        (self.m_squared + m_mode * m_mode / g_pp).sqrt()
    }

    fn nodal_proper_separation(&self) -> f64 {
        let g_pp = self.g_phiphi();
        (g_pp * self.dphi() * self.dphi()).sqrt().max(1e-12)
    }

    fn forward_proper_distance(&self, from: usize, to: usize) -> f64 {
        let n = self.n_nodes;
        let steps = (to + n - from) % n;
        if steps == 0 {
            return self.nodal_proper_separation();
        }
        let g_pp = self.g_phiphi();
        (g_pp * (self.dphi() * steps as f64).powi(2))
            .sqrt()
            .max(self.nodal_proper_separation())
    }

    /// Modular proxy sample that motivates \(\widetilde{K}\) (not a continuum Green function).
    fn proxy_sample(&self, delta_s: f64) -> f64 {
        let m = self.m_star();
        let ds = delta_s.max(1e-12);
        match self.kernel_kind {
            KernelKind::YukawaProxy => {
                (m / (4.0 * std::f64::consts::PI * ds)) * (-m * ds).exp()
            }
            KernelKind::ExponentialDecay => (-m * ds).exp(),
        }
    }

    /// Build the discrete kernel under the active [`RenormPolicy`].
    ///
    /// - [`RenormPolicy::None`]: physical proxy \(\widetilde{K}_{ij}=G(\Delta s)\,\Delta\tau\) (no row fix).
    /// - [`RenormPolicy::RowSumBenchmark`]: force row sums to \(\mathrm{e}^{-m_*\Delta\tau}\).
    pub fn build_kernel_matrix(&self) -> Vec<Vec<f64>> {
        let mut kernel = self.build_raw_kernel_matrix();
        if self.renorm == RenormPolicy::RowSumBenchmark {
            let g_pp = self.g_phiphi();
            let delta_tau = (g_pp * self.dphi() * self.dphi()).sqrt();
            let mass_gap = (-self.m_star() * delta_tau).exp();
            let n = self.n_nodes;
            for i in 0..n {
                let row_sum: f64 = kernel[i].iter().sum();
                if row_sum > 1e-15 {
                    let scale = mass_gap / row_sum;
                    for k in 0..n {
                        kernel[i][k] *= scale;
                    }
                }
            }
        }
        kernel
    }

    /// Unnormalized proxy kernel \(\widetilde{K}\) (always ignores [`RenormPolicy`]).
    pub fn build_raw_kernel_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_nodes;
        let g_pp = self.g_phiphi();
        let delta_tau = (g_pp * self.dphi() * self.dphi()).sqrt();
        let mut kernel = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let steps = (i + n - j) % n;
                if steps == 0 {
                    continue;
                }
                let ds = self.forward_proper_distance(j, i);
                kernel[i][j] = self.proxy_sample(ds) * delta_tau;
            }
        }
        kernel
    }

    /// Spectral radius of the **unnormalized** proxy \(\widetilde{K}\) (Path-3 diagnostic).
    pub fn unnormalized_spectral_radius(&self) -> f64 {
        spectral_radius_of(&self.build_raw_kernel_matrix())
    }

    pub fn apply(&self, field: &PeriodicField) -> PeriodicField {
        assert_eq!(
            field.len(),
            self.n_nodes,
            "field.len()={} must equal n_nodes={}",
            field.len(),
            self.n_nodes
        );
        PeriodicField {
            values: apply_kernel(&self.build_kernel_matrix(), &field.values),
        }
    }

    pub fn apply_with_kernel(&self, kernel: &[Vec<f64>], field: &PeriodicField) -> PeriodicField {
        assert_eq!(
            field.len(),
            self.n_nodes,
            "field.len()={} must equal n_nodes={}",
            field.len(),
            self.n_nodes
        );
        PeriodicField {
            values: apply_kernel(kernel, &field.values),
        }
    }

    /// Spectral-radius estimate of \(K\) via power iteration (dominant-eigenvalue magnitude).
    pub fn spectral_radius(&self) -> f64 {
        spectral_radius_of(&self.build_kernel_matrix())
    }

    /// Spectral-radius estimate used as the primary contraction diagnostic for \(K\).
    ///
    /// Historically named `l2_operator_norm`; this is \(\rho(K)\), not necessarily
    /// the induced \(L^2\) operator norm \(\sigma_{\max}(K)\).
    pub fn l2_operator_norm(&self) -> f64 {
        spectral_radius_of(&self.build_kernel_matrix())
    }

    /// Grid-oscillation Lipschitz diagnostic:
    /// \(L \le \rho(K)\,(1 + \pi\sqrt{2}/N)\).
    pub fn h1_operator_norm(&self) -> f64 {
        let rho = self.l2_operator_norm();
        let factor = 1.0 + std::f64::consts::PI * std::f64::consts::SQRT_2 / self.n_nodes as f64;
        rho * factor
    }

    pub fn lipschitz_bound(&self) -> f64 {
        self.h1_operator_norm()
    }

    pub fn is_contraction(&self) -> bool {
        self.m_squared > 0.0 && self.l2_operator_norm() < 1.0 && self.lipschitz_bound() < 1.0
    }
}

pub fn spectral_radius_of(kernel: &[Vec<f64>]) -> f64 {
    let n = kernel.len();
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let l2_norm = |x: &[f64]| (x.iter().map(|a| a * a).sum::<f64>() / n as f64).sqrt();
    let mut nv = l2_norm(&v).max(1e-15);
    for _ in 0..80 {
        v = apply_kernel(kernel, &v);
        nv = l2_norm(&v).max(1e-15);
        for x in &mut v {
            *x /= nv;
        }
    }
    nv
}

pub fn h1_operator_norm_of(kernel: &[Vec<f64>]) -> f64 {
    let n = kernel.len();
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let mut v_osc = oscillation_norm_slice(&v).max(1e-15);
    for x in &mut v {
        *x /= v_osc;
    }
    let mut lipschitz = 0.0;
    for _ in 0..80 {
        let kv = apply_kernel(kernel, &v);
        let kv_osc = oscillation_norm_slice(&kv);
        lipschitz = kv_osc / v_osc;
        v_osc = kv_osc.max(1e-15);
        v = kv.iter().map(|x| x / v_osc).collect();
    }
    lipschitz
}

/// Picard iteration \(\Phi\mapsto K\Phi + S\) with a pre-built kernel.
///
/// Homogeneous case \(S=0\) converges to the trivial fixed point \(\Phi^*=0\).
/// Nonzero forcing yields a nontrivial discrete response under \(\rho(K)<1\).
pub fn picard_iterate_with_kernel(
    initial: &PeriodicField,
    kernel: &[Vec<f64>],
    max_iter: usize,
    tol: f64,
) -> (PeriodicField, usize, f64) {
    let source = vec![0.0; kernel.len()];
    picard_iterate_inhomogeneous(initial, kernel, &source, max_iter, tol)
}

/// Inhomogeneous Picard map \(\Phi_{n+1} = K\Phi_n + S\).
pub fn picard_iterate_inhomogeneous(
    initial: &PeriodicField,
    kernel: &[Vec<f64>],
    source: &[f64],
    max_iter: usize,
    tol: f64,
) -> (PeriodicField, usize, f64) {
    assert_eq!(
        initial.len(),
        kernel.len(),
        "Picard: field.len()={} != kernel N={}",
        initial.len(),
        kernel.len()
    );
    assert_eq!(
        source.len(),
        kernel.len(),
        "Picard: source.len()={} != kernel N={}",
        source.len(),
        kernel.len()
    );
    let mut current = initial.clone();
    for k in 0..max_iter {
        let mapped = apply_kernel(kernel, &current.values);
        let next_vals: Vec<f64> = mapped
            .iter()
            .zip(source.iter())
            .map(|(kv, s)| kv + s)
            .collect();
        let next = PeriodicField { values: next_vals };
        let res = current.inhomogeneous_residual(kernel, source);
        if res < tol {
            return (next, k + 1, res);
        }
        current = next;
    }
    let res = current.inhomogeneous_residual(kernel, source);
    (current, max_iter, res)
}

pub fn picard_iterate(
    initial: &PeriodicField,
    m: &DiscreteMonodromyOperator,
    max_iter: usize,
    tol: f64,
) -> (PeriodicField, usize, f64) {
    assert_eq!(
        initial.len(),
        m.n_nodes,
        "Picard: field.len()={} != n_nodes={}",
        initial.len(),
        m.n_nodes
    );
    let kernel = m.build_kernel_matrix();
    picard_iterate_with_kernel(initial, &kernel, max_iter, tol)
}

pub fn find_fixed_point(
    initial: &PeriodicField,
    m: &DiscreteMonodromyOperator,
) -> Result<FixedPointResult> {
    let source = vec![0.0; m.n_nodes];
    find_fixed_point_inhomogeneous(initial, m, &source)
}

/// Solve the discrete inhomogeneous map \(\Phi = K\Phi + S\) by Picard iteration.
pub fn find_fixed_point_inhomogeneous(
    initial: &PeriodicField,
    m: &DiscreteMonodromyOperator,
    source: &[f64],
) -> Result<FixedPointResult> {
    assert_eq!(
        initial.len(),
        m.n_nodes,
        "fixed-point solve: field.len()={} must equal n_nodes={} (silent truncation forbidden)",
        initial.len(),
        m.n_nodes
    );
    assert_eq!(
        source.len(),
        m.n_nodes,
        "fixed-point solve: source.len()={} must equal n_nodes={}",
        source.len(),
        m.n_nodes
    );
    assert!(
        m.is_contraction(),
        "discrete monodromy M_N must be a contraction (m^2 > 0, L < 1)"
    );
    let kernel = m.build_kernel_matrix();
    let (field, iters, residual) =
        picard_iterate_inhomogeneous(initial, &kernel, source, 10_000, 1e-12);
    Ok(FixedPointResult {
        field,
        iterations: iters,
        residual,
        lipschitz: m.lipschitz_bound(),
    })
}

#[derive(Clone, Debug)]
pub struct FixedPointResult {
    pub field: PeriodicField,
    pub iterations: usize,
    pub residual: f64,
    pub lipschitz: f64,
}

pub fn banach_verification(m: &DiscreteMonodromyOperator) -> bool {
    m.is_contraction()
}

/// Default \(\mathcal{M}_N\) on the reference profile at \(r_{\mathrm{loop}}=1.5\,r_0\), \(N=128\).
///
/// Uses [`RenormPolicy::RowSumBenchmark`] so existing regression tests remain contractive.
pub fn default_monodromy_operator() -> DiscreteMonodromyOperator {
    use crate::algebra::metric_profiles::ExplicitProfileParams;
    let prof = ExplicitProfileParams::prd_reference();
    let r_loop = 1.5 * prof.r0;
    DiscreteMonodromyOperator {
        r_loop,
        params: prof.at(r_loop),
        m_squared: 1.0,
        azimuthal_mode: 1,
        n_nodes: 128,
        kernel_kind: KernelKind::YukawaProxy,
        renorm: RenormPolicy::RowSumBenchmark,
    }
}

/// Default operator with [`RenormPolicy::None`] (physical \(\widetilde{K}\), Path-3 stress tests).
pub fn default_unnormalized_monodromy_operator() -> DiscreteMonodromyOperator {
    let mut m = default_monodromy_operator();
    m.renorm = RenormPolicy::None;
    m
}

/// One sample of \(\rho(\widetilde{K}_N)\) on the Path-3 continuum grid.
#[derive(Clone, Debug)]
pub struct UnnormalizedSpectralSample {
    pub n_nodes: usize,
    pub m_squared: f64,
    /// Azimuthal mode \(\nu\) used in \(m_*=\sqrt{m^2+\nu^2/g_{\varphi\varphi}}\).
    pub azimuthal_mode: i32,
    pub kernel_kind: KernelKind,
    pub rho: f64,
    /// Mean absolute row sum of \(\widetilde{K}\) (proxy for \(\|G\|_{L^1}\) scale).
    pub mean_row_sum: f64,
    /// `true` iff \(\rho(\widetilde{K}_N) < 1\).
    pub contractive: bool,
}

impl UnnormalizedSpectralSample {
    pub fn ruptured(&self) -> bool {
        !self.contractive
    }
}

/// Configuration for an unnormalized Path-3 spectral sweep.
#[derive(Clone, Debug)]
pub struct UnnormalizedSweepConfig {
    pub n_values: Vec<usize>,
    pub m_squared_values: Vec<f64>,
    pub azimuthal_mode: i32,
    pub kernel_kind: KernelKind,
}

impl Default for UnnormalizedSweepConfig {
    fn default() -> Self {
        Self {
            n_values: path3_default_n_grid().to_vec(),
            m_squared_values: path3_default_m2_grid().to_vec(),
            azimuthal_mode: 1,
            kernel_kind: KernelKind::YukawaProxy,
        }
    }
}

/// Legacy continuum stress grid: \(N\in\{32,64,128,256\}\).
pub fn path3_default_n_grid() -> [usize; 4] {
    [32, 64, 128, 256]
}

/// High-resolution continuum grid including \(N\in\{512,1024\}\).
pub fn path3_extended_n_grid() -> [usize; 6] {
    [32, 64, 128, 256, 512, 1024]
}

/// Legacy bare-mass grid: \(m^2\in\{1,10^{-2},10^{-4}\}\).
pub fn path3_default_m2_grid() -> [f64; 3] {
    [1.0, 1e-2, 1e-4]
}

/// Ultra-low mass grid probing \(m^2\to 0\) (Coulomb / logarithmic regime when \(\nu=0\)).
pub fn path3_ultralow_m2_grid() -> [f64; 6] {
    [1.0, 1e-2, 1e-4, 1e-8, 1e-12, 0.0]
}

/// Path-3 high-res + \(\nu=0\) Coulomb/log configuration (exponential kernel).
pub fn path3_coulomb_sweep_config() -> UnnormalizedSweepConfig {
    UnnormalizedSweepConfig {
        n_values: path3_extended_n_grid().to_vec(),
        m_squared_values: path3_ultralow_m2_grid().to_vec(),
        azimuthal_mode: 0,
        kernel_kind: KernelKind::ExponentialDecay,
    }
}

/// Path-3 high-res + \(\nu=0\) with Yukawa proxy (prefactor \(\propto m_*\to 0\)).
pub fn path3_yukawa_nu0_sweep_config() -> UnnormalizedSweepConfig {
    UnnormalizedSweepConfig {
        n_values: path3_extended_n_grid().to_vec(),
        m_squared_values: path3_ultralow_m2_grid().to_vec(),
        azimuthal_mode: 0,
        kernel_kind: KernelKind::YukawaProxy,
    }
}

fn mean_abs_row_sum(kernel: &[Vec<f64>]) -> f64 {
    let n = kernel.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for row in kernel {
        acc += row.iter().map(|x| x.abs()).sum::<f64>();
    }
    acc / n as f64
}

/// Compute \(\rho(\widetilde{K}_N)\) for each pair \((N,m^2)\) on the unnormalized kernel.
///
/// Always assembles \(\widetilde{K}\) via [`DiscreteMonodromyOperator::build_raw_kernel_matrix`],
/// independent of the operator's [`RenormPolicy`] field. Uses default \(\nu=1\), Yukawa.
pub fn sweep_unnormalized_spectral_radius(
    n_values: &[usize],
    m_squared_values: &[f64],
) -> Vec<UnnormalizedSpectralSample> {
    sweep_unnormalized_spectral_radius_with(UnnormalizedSweepConfig {
        n_values: n_values.to_vec(),
        m_squared_values: m_squared_values.to_vec(),
        ..UnnormalizedSweepConfig::default()
    })
}

/// Full Path-3 sweep with configurable \(\nu\), kernel family, and grids.
pub fn sweep_unnormalized_spectral_radius_with(
    config: UnnormalizedSweepConfig,
) -> Vec<UnnormalizedSpectralSample> {
    let mut base = default_unnormalized_monodromy_operator();
    base.azimuthal_mode = config.azimuthal_mode;
    base.kernel_kind = config.kernel_kind;
    let mut out =
        Vec::with_capacity(config.n_values.len() * config.m_squared_values.len());
    for &m2 in &config.m_squared_values {
        for &n in &config.n_values {
            base.n_nodes = n;
            base.m_squared = m2;
            let kernel = base.build_raw_kernel_matrix();
            let rho = spectral_radius_of(&kernel);
            out.push(UnnormalizedSpectralSample {
                n_nodes: n,
                m_squared: m2,
                azimuthal_mode: config.azimuthal_mode,
                kernel_kind: config.kernel_kind,
                rho,
                mean_row_sum: mean_abs_row_sum(&kernel),
                contractive: rho < 1.0,
            });
        }
    }
    out
}

fn kernel_kind_label(kind: KernelKind) -> &'static str {
    match kind {
        KernelKind::YukawaProxy => "YukawaProxy",
        KernelKind::ExponentialDecay => "ExponentialDecay",
    }
}

/// Asymptotic / limit diagnostics extracted from a single \((m^2,\nu,\mathrm{kernel})\) series in \(N\).
#[derive(Clone, Debug)]
pub struct ContinuumLimitTrace {
    pub m_squared: f64,
    pub azimuthal_mode: i32,
    pub kernel_kind: KernelKind,
    pub rho_at_512: Option<f64>,
    pub rho_at_1024: Option<f64>,
    pub mean_row_sum_at_1024: Option<f64>,
    /// \(\rho(1024)-\rho(512)\) when both exist.
    pub delta_rho_512_to_1024: Option<f64>,
    /// Crude two-point extrapolation \(\rho_\infty \approx 2\rho_{1024}-\rho_{512}\) (constant step in \(N\to 2N\)).
    pub rho_infty_extrapolation: Option<f64>,
    pub crossed_unity: bool,
    pub verdict: String,
}

/// Build continuum-limit traces grouped by \((m^2,\nu,\mathrm{kernel})\).
pub fn continuum_limit_traces(samples: &[UnnormalizedSpectralSample]) -> Vec<ContinuumLimitTrace> {
    let mut keys: Vec<(u64, i32, KernelKind)> = Vec::new();
    for s in samples {
        let key = (s.m_squared.to_bits(), s.azimuthal_mode, s.kernel_kind);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    let mut traces = Vec::with_capacity(keys.len());
    for (m_bits, nu, kind) in keys {
        let m2 = f64::from_bits(m_bits);
        let series: Vec<&UnnormalizedSpectralSample> = samples
            .iter()
            .filter(|s| {
                s.m_squared.to_bits() == m_bits
                    && s.azimuthal_mode == nu
                    && s.kernel_kind == kind
            })
            .collect();
        let rho_at = |n: usize| series.iter().find(|s| s.n_nodes == n).map(|s| s.rho);
        let row_at = |n: usize| {
            series
                .iter()
                .find(|s| s.n_nodes == n)
                .map(|s| s.mean_row_sum)
        };
        let r512 = rho_at(512);
        let r1024 = rho_at(1024);
        let delta = match (r512, r1024) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        let extrap = match (r512, r1024) {
            (Some(a), Some(b)) => Some(2.0 * b - a),
            _ => None,
        };
        let crossed = series.iter().any(|s| s.ruptured());
        let verdict = match (r1024, delta, extrap, crossed) {
            (Some(r), _, _, true) if r >= 1.0 => {
                format!("RUPTURE at/before N=1024 (ρ={r:.6e} ≥ 1)")
            }
            (Some(r), Some(d), Some(e), false) if d > 0.0 && e >= 1.0 => {
                format!(
                    "growing toward rupture: ρ(1024)={r:.6e}, Δρ={d:.3e}, ρ_∞~{e:.6e} ≥ 1 (extrapolated)"
                )
            }
            (Some(r), Some(d), Some(e), false) if d.abs() < 1e-4 => {
                format!(
                    "asymptotically flat below unity: ρ(1024)={r:.6e}, |Δρ|={:.3e}, ρ_∞~{e:.6e}",
                    d.abs()
                )
            }
            (Some(r), Some(d), Some(e), false) if d > 0.0 && e < 1.0 => {
                format!(
                    "growing but extrapolated ρ_∞={e:.6e} < 1 (still contractive at continuum proxy)"
                )
            }
            (Some(r), Some(d), Some(e), false) if d < 0.0 => {
                format!(
                    "decreasing with N: ρ(1024)={r:.6e}, Δρ={d:.3e}, ρ_∞~{e:.6e}"
                )
            }
            (Some(r), _, _, false) => format!("contractive at N=1024 with ρ={r:.6e}"),
            _ => "incomplete high-N series (need N=512 and N=1024)".to_string(),
        };
        traces.push(ContinuumLimitTrace {
            m_squared: m2,
            azimuthal_mode: nu,
            kernel_kind: kind,
            rho_at_512: r512,
            rho_at_1024: r1024,
            mean_row_sum_at_1024: row_at(1024),
            delta_rho_512_to_1024: delta,
            rho_infty_extrapolation: extrap,
            crossed_unity: crossed,
            verdict,
        });
    }
    traces
}

/// Format a Path-3 sweep table (for CLI / `--nocapture` tests).
pub fn format_unnormalized_spectral_sweep(samples: &[UnnormalizedSpectralSample]) -> String {
    let nu = samples.first().map(|s| s.azimuthal_mode).unwrap_or(1);
    let kind = samples
        .first()
        .map(|s| kernel_kind_label(s.kernel_kind))
        .unwrap_or("?");
    let mut s = format!(
        "Path-3 unnormalized spectral sweep  ρ(K̃_N)\n\
         kernel={kind}  ν={nu}\n\
         =========================================================================\n\
         m^2          N      rho(K̃)     row_sum     contractive?\n\
         -------------------------------------------------------------------------\n"
    );
    for sample in samples {
        let flag = if sample.contractive {
            "yes (ρ<1)"
        } else {
            "NO  (ρ≥1)  ← rupture"
        };
        s.push_str(&format!(
            "{:<12.1e} {:<6} {:<10.6e} {:<10.6e} {}\n",
            sample.m_squared, sample.n_nodes, sample.rho, sample.mean_row_sum, flag
        ));
    }
    let n_rupture = samples.iter().filter(|x| x.ruptured()).count();
    s.push_str("-------------------------------------------------------------------------\n");
    s.push_str(&format!(
        "ruptures: {n_rupture} / {}\n",
        samples.len()
    ));

    let traces = continuum_limit_traces(samples);
    if !traces.is_empty() {
        s.push_str("\nContinuum-limit traces (N=512 → N=1024)\n");
        s.push_str("-------------------------------------------------------------------------\n");
        for t in &traces {
            s.push_str(&format!(
                "m²={:.1e}  ν={}  {}: {}\n",
                t.m_squared,
                t.azimuthal_mode,
                kernel_kind_label(t.kernel_kind),
                t.verdict
            ));
            if let (Some(r512), Some(r1024), Some(d), Some(e)) = (
                t.rho_at_512,
                t.rho_at_1024,
                t.delta_rho_512_to_1024,
                t.rho_infty_extrapolation,
            ) {
                s.push_str(&format!(
                    "    ρ(512)={r512:.6e}  ρ(1024)={r1024:.6e}  Δρ={d:.3e}  ρ_∞~{e:.6e}\n"
                ));
            }
        }
    }

    if n_rupture == 0 {
        s.push_str(
            "\nNo ρ≥1 on this grid: either the physical proxy remains contractive,\n\
             or the critical surface lies outside the probed (N,m²,ν,kernel) domain.\n",
        );
    } else {
        s.push_str(
            "\nCritical surface detected: unnormalized M_N loses contractivity on ≥1 sample.\n\
             RowSumBenchmark would have hidden these ruptures by construction.\n",
        );
    }
    s
}

/// Structured Markdown report for Path-3 high-resolution / Coulomb sweeps.
pub fn format_path3_continuum_report_markdown(
    yukawa_nu0: &[UnnormalizedSpectralSample],
    exponential_nu0: &[UnnormalizedSpectralSample],
) -> String {
    let mut md = String::from(
        "# Path-3 continuum stress report\n\n\
         Unnormalized discrete monodromy `K̃` (`RenormPolicy::None`).\n\n\
         - Profile: `ExplicitProfileParams::prd_reference()`, `r_loop = 1.5 r0`\n\
         - Grids: `N ∈ {32,64,128,256,512,1024}`, `m² ∈ {1, 1e-2, 1e-4, 1e-8, 1e-12, 0}`\n\
         - Azimuthal mode: `ν = 0` (no mode mass gap)\n\
         - Kernels: `YukawaProxy` vs `ExponentialDecay` (Coulomb / log probe)\n\n",
    );

    md.push_str("## 1. ExponentialDecay, ν = 0 (Coulomb / logarithmic regime)\n\n");
    md.push_str("```\n");
    md.push_str(&format_unnormalized_spectral_sweep(exponential_nu0));
    md.push_str("```\n\n");

    md.push_str("## 2. YukawaProxy, ν = 0 (prefactor ∝ m_* → 0)\n\n");
    md.push_str("```\n");
    md.push_str(&format_unnormalized_spectral_sweep(yukawa_nu0));
    md.push_str("```\n\n");

    md.push_str("## 3. Limit behavior at N = 1024\n\n");
    md.push_str("| kernel | m² | ρ(512) | ρ(1024) | Δρ | ρ_∞ (extrapolated) | row_sum(1024) | verdict |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---|\n");
    let mut all = exponential_nu0.to_vec();
    all.extend_from_slice(yukawa_nu0);
    for t in continuum_limit_traces(&all) {
        md.push_str(&format!(
            "| {} | {:.1e} | {} | {} | {} | {} | {} | {} |\n",
            kernel_kind_label(t.kernel_kind),
            t.m_squared,
            opt_f(t.rho_at_512),
            opt_f(t.rho_at_1024),
            opt_f(t.delta_rho_512_to_1024),
            opt_f(t.rho_infty_extrapolation),
            opt_f(t.mean_row_sum_at_1024),
            t.verdict.replace('|', "/"),
        ));
    }
    md.push('\n');

    md.push_str("## 4. CSV (machine-readable)\n\n");
    md.push_str("```csv\n");
    md.push_str("kernel,nu,m_squared,N,rho,mean_row_sum,contractive\n");
    for s in exponential_nu0.iter().chain(yukawa_nu0.iter()) {
        md.push_str(&format!(
            "{},{},{:.16e},{},{:.16e},{:.16e},{}\n",
            kernel_kind_label(s.kernel_kind),
            s.azimuthal_mode,
            s.m_squared,
            s.n_nodes,
            s.rho,
            s.mean_row_sum,
            s.contractive
        ));
    }
    md.push_str("```\n");
    md
}

fn opt_f(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{v:.6e}"),
        None => "—".to_string(),
    }
}

/// Write the Path-3 Markdown report to `path` (creates parent dirs as needed).
pub fn write_path3_continuum_report(
    path: impl AsRef<std::path::Path>,
    yukawa_nu0: &[UnnormalizedSpectralSample],
    exponential_nu0: &[UnnormalizedSpectralSample],
) -> std::io::Result<std::path::PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let md = format_path3_continuum_report_markdown(yukawa_nu0, exponential_nu0);
    std::fs::write(path, md)?;
    Ok(path.to_path_buf())
}

/// Run the extended Path-3 campaign and persist `docs/path3_continuum_report.md`.
pub fn run_path3_extended_campaign(
    report_path: impl AsRef<std::path::Path>,
) -> std::io::Result<(Vec<UnnormalizedSpectralSample>, Vec<UnnormalizedSpectralSample>, std::path::PathBuf)>
{
    let expo = sweep_unnormalized_spectral_radius_with(path3_coulomb_sweep_config());
    let yukawa = sweep_unnormalized_spectral_radius_with(path3_yukawa_nu0_sweep_config());
    let path = write_path3_continuum_report(report_path, &yukawa, &expo)?;
    Ok((yukawa, expo, path))
}

/// Unit-amplitude sinusoidal forcing on \(N\) nodes (default inhomogeneous workload).
pub fn default_forcing_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let chi = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            0.25 * chi.sin()
        })
        .collect()
}

fn sinusoidal_field(n: usize) -> PeriodicField {
    PeriodicField::new(
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin() + 1.5)
            .collect(),
    )
}

/// Homogeneous Picard regression on the default operator (converges to \(\Phi^*\approx 0\)).
pub fn homogeneous_picard_baseline_test() -> Result<FixedPointResult> {
    let m = default_monodromy_operator();
    let initial = sinusoidal_field(m.n_nodes);
    find_fixed_point(&initial, &m)
}

/// Inhomogeneous Picard regression: \(\Phi = K\Phi + S\) with nontrivial \(\|S\|_2>0\).
pub fn inhomogeneous_picard_baseline_test() -> Result<FixedPointResult> {
    let m = default_monodromy_operator();
    let initial = PeriodicField::new(vec![0.0; m.n_nodes]);
    let source = default_forcing_vector(m.n_nodes);
    find_fixed_point_inhomogeneous(&initial, &m, &source)
}

/// Comparative modular-kernel diagnostic: Yukawa vs exponential row-normalized \(\rho(K)\).
pub fn comparative_kernel_spectral_test() -> Result<(f64, f64)> {
    let mut yukawa = default_monodromy_operator();
    yukawa.n_nodes = 64;
    yukawa.kernel_kind = KernelKind::YukawaProxy;
    let mut expo = yukawa.clone();
    expo.kernel_kind = KernelKind::ExponentialDecay;
    let rho_y = yukawa.spectral_radius();
    let rho_e = expo.spectral_radius();
    assert!(rho_y < 1.0 && rho_e < 1.0);
    Ok((rho_y, rho_e))
}

pub fn alternate_initial_data_picard_test() -> Result<FixedPointResult> {
    let n = 64;
    let initial = PeriodicField::new(
        (0..n)
            .map(|i| {
                let phi = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (3.0 * phi).cos() + 2.0
            })
            .collect(),
    );
    let mut m = default_monodromy_operator();
    m.n_nodes = n;
    m.azimuthal_mode = 2;
    m.m_squared = 2.0;
    assert_eq!(initial.len(), m.n_nodes);
    find_fixed_point(&initial, &m)
}

/// Alias retained for older call sites that expected a non-homogeneous-sounding name.
pub fn nontrivial_fixed_point_test() -> Result<FixedPointResult> {
    inhomogeneous_picard_baseline_test()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_monodromy_is_contraction() {
        let m = default_monodromy_operator();
        assert!(m.m_squared > 0.0);
        assert!(m.is_contraction(), "L = {}", m.lipschitz_bound());
        assert!(banach_verification(&m));
    }

    #[test]
    fn homogeneous_picard_converges_to_trivial_fixed_point() {
        // Picard residuals for N ∈ {32, 64, 128} with exact dimension match.
        for &n in &[32usize, 64, 128] {
            let mut m = default_monodromy_operator();
            m.n_nodes = n;
            let init = PeriodicField::new(
                (0..n)
                    .map(|i| 1.0 + 0.1 * (i as f64) + 0.05 * (i as f64 * 0.2).sin())
                    .collect(),
            );
            assert_eq!(init.len(), m.n_nodes);
            let result = find_fixed_point(&init, &m).unwrap();
            assert!(
                result.residual < 1e-10,
                "N={n}: residual {:.3e} not < 1e-10",
                result.residual
            );
            assert!(result.lipschitz < 1.0, "N={n}: L = {}", result.lipschitz);
            assert!(
                result.iterations < 500,
                "N={n}: iters = {}",
                result.iterations
            );
            // Homogeneous contraction → trivial fixed point.
            assert!(
                result.field.l2_norm() < 1e-6,
                "N={n}: expected ||Φ*||≈0, got {}",
                result.field.l2_norm()
            );
        }
    }

    #[test]
    fn inhomogeneous_forcing_yields_nontrivial_response() {
        let mut m = default_monodromy_operator();
        m.n_nodes = 64;
        let source = default_forcing_vector(m.n_nodes);
        let s_norm = (source.iter().map(|s| s * s).sum::<f64>() / m.n_nodes as f64).sqrt();
        assert!(s_norm > 1e-3, "forcing must be nontrivial");
        let init = PeriodicField::new(vec![0.0; m.n_nodes]);
        let result = find_fixed_point_inhomogeneous(&init, &m, &source).unwrap();
        assert!(result.residual < 1e-10, "residual = {:.3e}", result.residual);
        assert!(
            result.field.l2_norm() > 1e-3,
            "expected nontrivial ||Φ*||, got {}",
            result.field.l2_norm()
        );
        assert!(result.iterations < 500);
        // Comparative modular kernels both remain contractions.
        let (rho_y, rho_e) = comparative_kernel_spectral_test().unwrap();
        assert!(rho_y < 1.0 && rho_e < 1.0);
    }

    #[test]
    fn oscillation_proxy_contracts_under_kernel() {
        let m = default_monodromy_operator();
        let kernel = m.build_kernel_matrix();
        assert_eq!(kernel.len(), m.n_nodes);
        let rho = m.l2_operator_norm();
        let l_osc = m.h1_operator_norm();
        assert!(rho < 1.0, "spectral-radius diagnostic = {rho}");
        assert!(l_osc < 1.0, "oscillation-proxy bound = {l_osc}");
        let init = PeriodicField::new((0..m.n_nodes).map(|i| (i as f64 * 0.3).sin()).collect());
        assert_eq!(init.len(), m.n_nodes);
        let out_vals = apply_kernel(&kernel, &init.values);
        assert_eq!(out_vals.len(), m.n_nodes);
        assert!(
            out_vals.iter().map(|v| v * v).sum::<f64>()
                <= rho * rho * init.values.iter().map(|v| v * v).sum::<f64>() + 1e-8
        );
    }

    #[test]
    fn renorm_policy_none_skips_row_sum_fix() {
        let mut m = default_monodromy_operator();
        m.n_nodes = 32;
        m.renorm = RenormPolicy::RowSumBenchmark;
        let k_bench = m.build_kernel_matrix();
        let g_pp = {
            let (g, _) = crate::algebra::metric_numeric::metric_at(
                m.r_loop,
                std::f64::consts::FRAC_PI_2,
                &m.params,
            );
            g[3][3]
        };
        let dphi = 2.0 * std::f64::consts::PI / m.n_nodes as f64;
        let delta_tau = (g_pp * dphi * dphi).sqrt();
        let expected_row = (-m.m_star() * delta_tau).exp();
        for row in &k_bench {
            let sum: f64 = row.iter().sum();
            assert!((sum - expected_row).abs() < 1e-10, "row sum {sum} vs {expected_row}");
        }

        m.renorm = RenormPolicy::None;
        let k_phys = m.build_kernel_matrix();
        let k_raw = m.build_raw_kernel_matrix();
        for i in 0..m.n_nodes {
            for j in 0..m.n_nodes {
                assert!((k_phys[i][j] - k_raw[i][j]).abs() < 1e-15);
            }
        }
        // Physical kernel must differ from the forced row-sum benchmark (artifact removal).
        let mut max_diff: f64 = 0.0;
        for i in 0..m.n_nodes {
            for j in 0..m.n_nodes {
                max_diff = max_diff.max((k_phys[i][j] - k_bench[i][j]).abs());
            }
        }
        assert!(max_diff > 1e-6, "expected physical vs benchmark kernels to differ");
    }

    #[test]
    fn path3_unnormalized_spectral_sweep_reports_contractivity() {
        let samples =
            sweep_unnormalized_spectral_radius(&path3_default_n_grid(), &path3_default_m2_grid());
        assert_eq!(samples.len(), 4 * 3);
        let report = format_unnormalized_spectral_sweep(&samples);
        println!("\n{report}");

        // Benchmark policy must remain contractive on the same (N, m²) corners.
        for sample in &samples {
            let mut m = default_monodromy_operator();
            m.n_nodes = sample.n_nodes;
            m.m_squared = sample.m_squared;
            m.renorm = RenormPolicy::RowSumBenchmark;
            assert!(
                m.spectral_radius() < 1.0,
                "RowSumBenchmark unexpectedly non-contractive at N={}, m²={}",
                sample.n_nodes,
                sample.m_squared
            );
        }

        // Sanity: ρ is finite and nonnegative; document ruptures without forcing them.
        for sample in &samples {
            assert!(sample.rho.is_finite() && sample.rho >= 0.0);
            assert_eq!(sample.contractive, sample.rho < 1.0);
            if sample.ruptured() {
                println!(
                    "RUPTURE: N={}, m²={:.1e}, ρ={:.6e}",
                    sample.n_nodes, sample.m_squared, sample.rho
                );
            }
        }
    }
}
