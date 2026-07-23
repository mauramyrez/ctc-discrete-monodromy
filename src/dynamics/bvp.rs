//! Compatibility re-exports for the discrete monodromy operator \(\mathcal{M}_N\).
//!
//! Prefer `crate::dynamics::monodromy` for new call sites. Certified claims refer
//! to the row-sum renormalized kernel \(K\) and the discrete grid-oscillation proxy.

pub use super::monodromy::{
    banach_verification, find_fixed_point, find_fixed_point_inhomogeneous,
    homogeneous_picard_baseline_test, inhomogeneous_picard_baseline_test, FixedPointResult,
    KernelKind, MonodromyOperator, PeriodicField,
};

use anyhow::Result;

/// Smoke-test wrapper for homogeneous Picard convergence of \(\mathcal{M}_N\).
pub fn homogeneous_picard_smoke_test() -> Result<&'static str> {
    let fp = homogeneous_picard_baseline_test()?;
    if fp.residual < 1e-8 && fp.field.l2_norm() < 1e-6 {
        Ok("PASS (homogeneous M_N Picard → Φ*≈0; spectral regression)")
    } else {
        Ok("FAIL (residual or ||Φ*|| above tolerance)")
    }
}

/// Smoke-test wrapper for the inhomogeneous map \(\Phi = K\Phi + S\).
pub fn inhomogeneous_picard_smoke_test() -> Result<&'static str> {
    let fp = inhomogeneous_picard_baseline_test()?;
    if fp.residual < 1e-8 && fp.field.l2_norm() > 1e-3 {
        Ok("PASS (inhomogeneous M_N Picard → nontrivial Φ*; discrete response)")
    } else {
        Ok("FAIL (residual or ||Φ*|| outside tolerance)")
    }
}
