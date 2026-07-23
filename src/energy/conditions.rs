//! Classical energy-condition API scaffold (criterion C2 companion).
//!
//! These unit tests exercise only quadratic-form sanity checks on sample
//! tensors. They do **not** assemble \(G_{\mu\nu}\) dynamically from the Rust
//! curvature engine; SymPy exports supply the manuscript's background
//! NEC/WEC constraints.
//!
//! - NEC: \(T_{\mu\nu} k^\mu k^\nu \ge 0\) for null \(k^\mu\)
//! - WEC: \(T_{\mu\nu} u^\mu u^\nu \ge 0\) for timelike \(u^\mu\)
//! - SEC: \((T_{\mu\nu} - \tfrac12 T g_{\mu\nu}) u^\mu u^\nu \ge 0\) (hook only)

use super::stress_energy::{quadratic_form, sample_ordinary_stress_energy};
use anyhow::Result;

/// Null vector sample \(k^\mu = (1, 1, 0, 0)\) in Minkowski (−+++) — null when \(\eta(k,k)=0\).
fn sample_null() -> [f64; 4] {
    [1.0, 1.0, 0.0, 0.0]
}

/// Future-directed unit timelike sample \(u^\mu = (1, 0, 0, 0)\).
fn sample_timelike() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

pub fn null_energy_condition_sample() -> Result<bool> {
    let t = sample_ordinary_stress_energy();
    let k = sample_null();
    Ok(quadratic_form(&t, &k) >= 0.0)
}

pub fn weak_energy_condition_sample() -> Result<bool> {
    let t = sample_ordinary_stress_energy();
    let u = sample_timelike();
    Ok(quadratic_form(&t, &u) >= 0.0)
}

/// Region classifier: returns true if NEC holds for the provided \(T\) and null cone samples.
pub fn nec_holds_on_samples(t: &nalgebra::Matrix4<f64>, null_rays: &[[f64; 4]]) -> bool {
    null_rays.iter().all(|k| quadratic_form(t, k) >= -1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::stress_energy::sample_exotic_stress_energy;

    #[test]
    fn ordinary_matter_passes_wec_sample() {
        assert!(weak_energy_condition_sample().unwrap());
    }

    #[test]
    fn exotic_sample_flagged() {
        let t = sample_exotic_stress_energy();
        let u = sample_timelike();
        // Scaffold T_00 < 0 → WEC violation for u = ∂_t
        assert!(quadratic_form(&t, &u) < 0.0);
    }
}
