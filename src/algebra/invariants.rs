//! Curvature / metric invariant reports for high-precision stress tests.

use super::metric::{evaluate_metric_slice, metric_determinant};
use anyhow::Result;
use num_rational::BigRational;
use num_traits::{One, Zero};

/// Sample metric determinant and redshift factor e^{2Φ}.
pub fn sample_metric_report() -> Result<String> {
    let r = BigRational::from_integer(3.into());
    let b_over_r = BigRational::new(1.into(), 3.into());
    let phi = BigRational::new((-1).into(), 20.into());
    let omega = BigRational::new(1.into(), 5.into());
    let sin_theta = BigRational::one();

    let g = evaluate_metric_slice(r, b_over_r, phi.clone(), omega, sin_theta);
    let det = metric_determinant(&g);

    let e2phi_note = redshift_factor_note(&phi);

    Ok(format!(
        "det(g) = {det}; {e2phi_note}; Φ scaffold exact={phi}"
    ))
}

fn redshift_factor_note(phi: &BigRational) -> String {
    #[cfg(feature = "high-prec")]
    {
        use rug::Float;
        const PREC: u32 = 256;
        let phi_f = Float::with_val(PREC, &*format!("{}", phi));
        let e2phi = (Float::with_val(PREC, 2) * phi_f).exp();
        format!("e^{{2Φ}} ≈ {e2phi} (rug, {PREC} bits)")
    }
    #[cfg(not(feature = "high-prec"))]
    {
        // Series: e^{2Φ} ≈ 1 + 2Φ + 2Φ² (scaffold without rug)
        let two = BigRational::from_integer(2.into());
        let approx = BigRational::one() + &two * phi + &two * phi * phi;
        format!("e^{{2Φ}} ≈ {approx} (series; enable --features high-prec for MPFR)")
    }
}

/// Scalar curvature placeholder (filled once Riemann pipeline is wired).
pub fn ricci_scalar_placeholder() -> BigRational {
    BigRational::zero()
}
