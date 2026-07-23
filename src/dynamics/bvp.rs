//! Compatibility re-exports for the discrete monodromy operator \(\mathcal{M}_N\).
//!
//! Prefer `crate::dynamics::novikov` for new call sites. Certified claims refer
//! to the row-sum renormalized kernel \(K\) and the discrete \(H^1\)-proxy norm.

pub use super::novikov::{
    banach_verification, find_fixed_point, novikov_fixed_point_test, FixedPointResult,
    MonodromyOperator, PeriodicField,
};

use anyhow::Result;

/// Smoke-test wrapper for discrete Banach–Picard convergence of \(\mathcal{M}_N\).
pub fn novikov_fixed_point_smoke_test() -> Result<&'static str> {
    let fp = novikov_fixed_point_test()?;
    if fp.residual < 1e-8 {
        Ok("PASS (discrete M_N fixed point; Banach–Picard contraction)")
    } else {
        Ok("FAIL (residual above tolerance)")
    }
}
