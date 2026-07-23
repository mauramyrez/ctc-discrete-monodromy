//! Generalized Morris–Thorne–like metric with azimuthal frame-dragging.
//!
//! Line element (signature −+++):
//! ```text
//! ds² = −e^{2Φ(r)} dt² + dr²/(1−b(r)/r) + r² dθ² + r² sin²θ [dφ − ω(r) dt]²
//! ```
//! The shift ω(r) generates frame dragging; when |ω| is large enough that
//! g_tt > 0 on an equatorial circle of fixed (t,r,θ), an ergoregion appears.

use num_rational::BigRational;
use num_traits::{One, Zero};

/// Coordinate labels: (t, r, θ, φ) ↔ (0, 1, 2, 3).
pub const DIM: usize = 4;

/// Exact rational 4×4 matrix (row-major) in (−+++) signature.
pub type MetricRational = [[BigRational; DIM]; DIM];

fn zero_metric() -> MetricRational {
    std::array::from_fn(|_| std::array::from_fn(|_| BigRational::zero()))
}

/// Placeholder evaluation of a static, equatorial slice of a frame-dragging metric.
///
/// Parameters (geometric units):
/// - `r`: radial coordinate
/// - `b_over_r`: shape function ratio b(r)/r ∈ [0, 1)
/// - `phi`: redshift Φ(r)
/// - `omega`: frame-dragging ω(r)
/// - `sin_theta`: sin(θ) (1 for equatorial)
pub fn evaluate_metric_slice(
    r: BigRational,
    b_over_r: BigRational,
    phi: BigRational,
    omega: BigRational,
    sin_theta: BigRational,
) -> MetricRational {
    let r2 = &r * &r;
    let r2_sin2 = &r2 * &sin_theta * &sin_theta;

    // Approximate e^{2Φ} ≈ 1 + 2Φ for small Φ (exact exp via rug in invariants).
    let e2phi = BigRational::one() + BigRational::from_integer(2.into()) * &phi;

    let g_tt = -&e2phi + &r2_sin2 * &omega * &omega;
    let g_rr = BigRational::one() / (BigRational::one() - b_over_r);
    let g_thth = r2;
    let g_phph = r2_sin2.clone();
    let g_tphi = -&r2_sin2 * &omega;

    let mut g = zero_metric();
    g[0][0] = g_tt;
    g[1][1] = g_rr;
    g[2][2] = g_thth;
    g[3][3] = g_phph;
    g[0][3] = g_tphi.clone();
    g[3][0] = g_tphi;
    g
}

/// 4×4 determinant via Laplace expansion (exact rationals).
pub fn metric_determinant(g: &MetricRational) -> BigRational {
    fn det3(m: &[[BigRational; 3]; 3]) -> BigRational {
        &m[0][0] * (&m[1][1] * &m[2][2] - &m[1][2] * &m[2][1])
            - &m[0][1] * (&m[1][0] * &m[2][2] - &m[1][2] * &m[2][0])
            + &m[0][2] * (&m[1][0] * &m[2][1] - &m[1][1] * &m[2][0])
    }

    let mut acc = BigRational::zero();
    for j in 0..DIM {
        let mut minor = std::array::from_fn(|_| std::array::from_fn(|_| BigRational::zero()));
        let mut r = 0;
        for i in 1..DIM {
            let mut c = 0;
            for k in 0..DIM {
                if k == j {
                    continue;
                }
                minor[r][c] = g[i][k].clone();
                c += 1;
            }
            r += 1;
        }
        let sign = if j % 2 == 0 {
            BigRational::one()
        } else {
            -BigRational::one()
        };
        acc += sign * &g[0][j] * det3(&minor);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{Signed, ToPrimitive};

    #[test]
    fn equatorial_slice_has_lorentzian_det_sign() {
        let r = BigRational::from_integer(2.into());
        let b_over_r = BigRational::new(1.into(), 2.into());
        let phi = BigRational::zero();
        let omega = BigRational::new(1.into(), 10.into());
        let sin_theta = BigRational::one();
        let g = evaluate_metric_slice(r, b_over_r, phi, omega, sin_theta);
        let det = metric_determinant(&g);
        assert!(det.to_f64().unwrap() < 0.0);
        assert!(det.is_negative());
    }
}
