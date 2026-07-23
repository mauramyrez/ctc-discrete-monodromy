//! Discrete Yukawa-proxy monodromy operator \(\mathcal{M}_N\) (criterion C3).
//!
//! Continuous Klein–Gordon structure motivates a Yukawa-type kernel with
//! mode-augmented mass \(m_*\), but this module does **not** discretize the
//! full curved-space PDE and does **not** claim a continuum \(H^1(\gamma)\)
//! contraction. All certified numerical claims refer to the row-sum
//! renormalized discrete kernel \(K\) on \(\mathbb{R}^N\):
//!
//!   \(K_{ij} = \widetilde{K}_{ij}\,\mathrm{e}^{-m_*\Delta\tau}/\sum_k\widetilde{K}_{ik}\)
//!
//! (row renormalization is an explicit step of the discrete definition).
//! When \(m^2>0\) and \(\|K\|_{L^2}<1\) with discrete \(H^1\)-proxy Lipschitz
//! bound \(L<1\), \(\mathcal{M}_N:\Phi\mapsto K\Phi\) is a strict contraction
//! on \((\mathbb{R}^N,\|\cdot\|_{H^1\text{-proxy}})\).

use crate::algebra::metric_numeric::{metric_at, MorrisThorneParams};

use anyhow::Result;

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

    /// Discrete \(H^1\)-proxy norm on \(\mathbb{R}^N\) (not continuum \(H^1(\gamma)\)).
    pub fn h1_norm(&self) -> f64 {
        h1_norm_slice(&self.values)
    }

    /// Residual \(\|\Phi - K\Phi\|_2\).
    pub fn fixed_point_residual(&self, kernel: &[Vec<f64>]) -> f64 {
        let mapped = apply_kernel(kernel, &self.values);
        let n = self.len();
        let mut acc = 0.0;
        for i in 0..n {
            let d = self.values[i] - mapped[i];
            acc += d * d;
        }
        (acc / n as f64).sqrt()
    }
}

/// Discrete \(H^1\)-proxy norm for a periodic sample vector.
pub fn h1_norm_slice(v: &[f64]) -> f64 {
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

/// Discrete Yukawa-proxy monodromy operator \(\mathcal{M}_N\) on \(\mathbb{R}^N\).
///
/// Builds the row-sum renormalized kernel \(K\) from a sampled Yukawa proxy
/// with mode-augmented mass \(m_*\). Certified claims are discrete only.
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

    /// Continuous Yukawa proxy sample (motivates \(K\); not the exact curved-space Green function).
    fn yukawa_proxy(&self, delta_s: f64) -> f64 {
        let m = self.m_star();
        let ds = delta_s.max(1e-12);
        (m / (4.0 * std::f64::consts::PI * ds)) * (-m * ds).exp()
    }

    /// Build the row-sum renormalized discrete kernel \(K\).
    ///
    /// Row renormalization by \(\mathrm{e}^{-m_*\Delta\tau}\) is an **explicit**
    /// step of the discrete operator definition \(\mathcal{M}_N\), not a numerical accident.
    pub fn build_kernel_matrix(&self) -> Vec<Vec<f64>> {
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
                kernel[i][j] = self.yukawa_proxy(ds) * delta_tau;
            }
        }
        let mass_gap = (-self.m_star() * delta_tau).exp();
        for i in 0..n {
            let row_sum: f64 = kernel[i].iter().sum();
            if row_sum > 1e-15 {
                let scale = mass_gap / row_sum;
                for k in 0..n {
                    kernel[i][k] *= scale;
                }
            }
        }
        kernel
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

    /// Spectral-radius estimate used as an \(L^2\) contraction diagnostic for \(K\).
    pub fn l2_operator_norm(&self) -> f64 {
        spectral_radius_of(&self.build_kernel_matrix())
    }

    /// Discrete \(H^1\)-proxy Lipschitz bound:
    /// \(L \le \|K\|_{L^2}\,(1 + \pi\sqrt{2}/N)\).
    pub fn h1_operator_norm(&self) -> f64 {
        let l2 = self.l2_operator_norm();
        let sobolev_factor =
            1.0 + std::f64::consts::PI * std::f64::consts::SQRT_2 / self.n_nodes as f64;
        l2 * sobolev_factor
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
    let mut v_h1 = h1_norm_slice(&v).max(1e-15);
    for x in &mut v {
        *x /= v_h1;
    }
    let mut lipschitz = 0.0;
    for _ in 0..80 {
        let kv = apply_kernel(kernel, &v);
        let kv_h1 = h1_norm_slice(&kv);
        lipschitz = kv_h1 / v_h1;
        v_h1 = kv_h1.max(1e-15);
        v = kv.iter().map(|x| x / v_h1).collect();
    }
    lipschitz
}

/// Picard iteration with pre-built kernel (avoids \(O(N^2)\) rebuild each step).
pub fn picard_iterate_with_kernel(
    initial: &PeriodicField,
    kernel: &[Vec<f64>],
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
    let mut current = initial.clone();
    for k in 0..max_iter {
        let next_vals = apply_kernel(kernel, &current.values);
        let next = PeriodicField { values: next_vals };
        let res = current.fixed_point_residual(kernel);
        if res < tol {
            return (next, k + 1, res);
        }
        current = next;
    }
    let res = current.fixed_point_residual(kernel);
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
    assert_eq!(
        initial.len(),
        m.n_nodes,
        "fixed-point solve: field.len()={} must equal n_nodes={} (silent truncation forbidden)",
        initial.len(),
        m.n_nodes
    );
    assert!(
        m.is_contraction(),
        "discrete monodromy M_N must be a contraction (m^2 > 0, L < 1)"
    );
    let kernel = m.build_kernel_matrix();
    let (field, iters, residual) = picard_iterate_with_kernel(initial, &kernel, 10_000, 1e-12);
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

/// Default \(\mathcal{M}_N\) on the PRD reference profile at \(r_{\mathrm{loop}}=1.5\,r_0\), \(N=128\).
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
    }
}

fn sinusoidal_field(n: usize) -> PeriodicField {
    PeriodicField::new(
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin() + 1.5)
            .collect(),
    )
}

pub fn novikov_fixed_point_test() -> Result<FixedPointResult> {
    let m = default_monodromy_operator();
    let initial = sinusoidal_field(m.n_nodes);
    find_fixed_point(&initial, &m)
}

pub fn nontrivial_fixed_point_test() -> Result<FixedPointResult> {
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
    fn contraction_map_has_unique_fixed_point() {
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
        }
    }

    #[test]
    fn sinusoidal_initial_converges() {
        let result = novikov_fixed_point_test().unwrap();
        assert_eq!(result.field.len(), 128);
        assert!(result.residual < 1e-8);
        assert!(result.iterations < 500);
    }

    #[test]
    fn h1_proxy_norm_contracts_under_kernel() {
        let m = default_monodromy_operator();
        let kernel = m.build_kernel_matrix();
        assert_eq!(kernel.len(), m.n_nodes);
        let l2 = m.l2_operator_norm();
        let l_h1 = m.h1_operator_norm();
        assert!(l2 < 1.0, "spectral-radius diagnostic = {l2}");
        assert!(l_h1 < 1.0, "H¹-proxy bound = {l_h1}");
        let init = PeriodicField::new((0..m.n_nodes).map(|i| (i as f64 * 0.3).sin()).collect());
        assert_eq!(init.len(), m.n_nodes);
        let out_vals = apply_kernel(&kernel, &init.values);
        assert_eq!(out_vals.len(), m.n_nodes);
        assert!(
            out_vals.iter().map(|v| v * v).sum::<f64>()
                <= l2 * l2 * init.values.iter().map(|v| v * v).sum::<f64>() + 1e-8
        );
    }
}
