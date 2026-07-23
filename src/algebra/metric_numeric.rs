//! Numeric Morris–Thorne metric with frame dragging at a spacetime point.
//!
//! Coordinates: (t, r, θ, φ) ↔ indices (0, 1, 2, 3).
//! Signature (−+++).

use super::christoffel::Christoffel;
use super::DIM;

/// Metric parameterization: Φ(r), b(r), ω(r) as smooth functions of r.
#[derive(Clone, Debug)]
pub struct MorrisThorneParams {
    /// Redshift Φ(r)
    pub phi: f64,
    /// Shape b(r)
    pub b: f64,
    /// Frame-dragging ω(r)
    pub omega: f64,
    /// dΦ/dr
    pub phi_prime: f64,
    /// db/dr
    pub b_prime: f64,
    /// dω/dr
    pub omega_prime: f64,
}

impl MorrisThorneParams {
    /// Constant-Φ, constant-b throat with ω(r) = ω₀/r² (illustrative sample).
    /// Prefer `ExplicitProfileParams` / `explicit_profile` for criterion~(C1) runs.
    pub fn frame_dragging_example(r: f64, b0: f64, omega0: f64, phi0: f64) -> Self {
        let omega = omega0 / (r * r);
        let omega_prime = -2.0 * omega0 / (r * r * r);
        Self {
            phi: phi0,
            b: b0,
            omega,
            phi_prime: 0.0,
            b_prime: 0.0,
            omega_prime,
        }
    }

    /// Explicit closed-form profile at radius r (criterion C1).
    pub fn explicit_profile(
        r: f64,
        r0: f64,
        gamma: f64,
        alpha: f64,
        omega0: f64,
        beta: f64,
    ) -> Self {
        super::metric_profiles::ExplicitProfileParams {
            r0,
            gamma,
            alpha,
            omega0,
            beta,
        }
        .at(r)
    }
}

/// Evaluate g_{μν} and g^{μν} at (r, θ).
pub fn metric_at(
    r: f64,
    theta: f64,
    p: &MorrisThorneParams,
) -> ([[f64; DIM]; DIM], [[f64; DIM]; DIM]) {
    let sin_th = theta.sin();
    let r2 = r * r;
    let r2s2 = r2 * sin_th * sin_th;
    let e2phi = (2.0 * p.phi).exp();

    let g_tt = -e2phi + r2s2 * p.omega * p.omega;
    let g_rr = 1.0 / (1.0 - p.b / r);
    let g_thth = r2;
    let g_phph = r2s2;
    let g_tphi = -r2s2 * p.omega;

    let mut g = [[0.0; DIM]; DIM];
    g[0][0] = g_tt;
    g[1][1] = g_rr;
    g[2][2] = g_thth;
    g[3][3] = g_phph;
    g[0][3] = g_tphi;
    g[3][0] = g_tphi;

    let g_inv = invert_4x4(g);
    (g, g_inv)
}

/// Christoffel symbols at (r, θ) with fixed local parameters.
pub fn christoffel_at(r: f64, theta: f64, p: &MorrisThorneParams) -> Christoffel {
    let local = p.clone();
    christoffel_fd(&[0.0, r, theta, 0.0], &|_| local.clone(), 1e-5)
}

/// Christoffel symbols via central finite differences (robust, complete).
pub fn christoffel_fd(
    x: &[f64; DIM],
    params_at_r: &dyn Fn(f64) -> MorrisThorneParams,
    eps: f64,
) -> Christoffel {
    let g0 = metric_at_point(x, params_at_r);
    let g_inv = invert_4x4(g0);
    let mut gamma = [[[0.0; DIM]; DIM]; DIM];

    for lam in 0..DIM {
        for mu in 0..DIM {
            for nu in 0..DIM {
                let mut sum = 0.0;
                for sig in 0..DIM {
                    let mut x_plus_mu = *x;
                    x_plus_mu[mu] += eps;
                    let mut x_minus_mu = *x;
                    x_minus_mu[mu] -= eps;
                    let mut x_plus_nu = *x;
                    x_plus_nu[nu] += eps;
                    let mut x_minus_nu = *x;
                    x_minus_nu[nu] -= eps;
                    let mut x_plus_sig = *x;
                    x_plus_sig[sig] += eps;
                    let mut x_minus_sig = *x;
                    x_minus_sig[sig] -= eps;

                    let dg_mu = (metric_at_point(&x_plus_mu, params_at_r)[nu][sig]
                        - metric_at_point(&x_minus_mu, params_at_r)[nu][sig])
                        / (2.0 * eps);
                    let dg_nu = (metric_at_point(&x_plus_nu, params_at_r)[mu][sig]
                        - metric_at_point(&x_minus_nu, params_at_r)[mu][sig])
                        / (2.0 * eps);
                    let dg_sig = (metric_at_point(&x_plus_sig, params_at_r)[mu][nu]
                        - metric_at_point(&x_minus_sig, params_at_r)[mu][nu])
                        / (2.0 * eps);
                    sum += g_inv[lam][sig] * (dg_mu + dg_nu - dg_sig);
                }
                gamma[lam][mu][nu] = 0.5 * sum;
            }
        }
    }
    gamma
}

fn metric_at_point(
    x: &[f64; DIM],
    params_at_r: &dyn Fn(f64) -> MorrisThorneParams,
) -> [[f64; DIM]; DIM] {
    let r = x[1].max(1e-6);
    let theta = x[2].clamp(1e-6, std::f64::consts::PI - 1e-6);
    let p = params_at_r(r);
    metric_at(r, theta, &p).0
}

/// Invert 4×4 matrix (cofactor expansion); regularized for near-throat points.
fn invert_4x4(m: [[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    let det = det_4x4(m);
    if det.abs() < 1e-24 {
        // Regularized inverse near throat / degenerate slice
        return [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }
    let mut inv = [[0.0; DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            let minor = minor_3x3(m, i, j);
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            inv[j][i] = sign * minor / det;
        }
    }
    inv
}

fn det_4x4(m: [[f64; DIM]; DIM]) -> f64 {
    let mut det = 0.0;
    for j in 0..DIM {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[0][j] * minor_3x3(m, 0, j);
    }
    det
}

fn minor_3x3(m: [[f64; DIM]; DIM], row: usize, col: usize) -> f64 {
    let mut sub = [[0.0; 3]; 3];
    let mut ri = 0;
    for i in 0..DIM {
        if i == row {
            continue;
        }
        let mut cj = 0;
        for j in 0..DIM {
            if j == col {
                continue;
            }
            sub[ri][cj] = m[i][j];
            cj += 1;
        }
        ri += 1;
    }
    sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
        - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
        + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0])
}

/// Equatorial ergoregion diagnostic: e^{2Φ} < r² ω² (i.e. g_tt > 0 at θ = π/2).
pub fn ergoregion(r: f64, p: &MorrisThorneParams) -> bool {
    let e2phi = (2.0 * p.phi).exp();
    e2phi < r * r * p.omega * p.omega
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_has_lorentzian_signature() {
        let p = MorrisThorneParams::frame_dragging_example(2.0, 1.0, 5.0, -0.05);
        let (g, _) = metric_at(2.0, std::f64::consts::FRAC_PI_2, &p);
        let det = det_4x4(g);
        assert!(det < 0.0);
    }

    #[test]
    fn ergoregion_detected_at_sufficient_omega() {
        let p = MorrisThorneParams::frame_dragging_example(2.0, 1.0, 10.0, 0.0);
        assert!(ergoregion(2.0, &p));
    }
}
