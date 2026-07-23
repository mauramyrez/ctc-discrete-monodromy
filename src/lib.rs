//! High-performance Rust framework for discrete operator diagnostics on
//! axisymmetric background profiles.
//!
//! Signature: (−+++). Geometric units \(c = G = 1\).
//!
//! Pipeline capabilities:
//! - Explicit frame-dragging Morris–Thorne profiles and equatorial ergoregion locus.
//! - SymPy-exported equatorial Einstein / energy-condition assets (static; not Rust-dynamical).
//! - Discrete monodromy \(\mathcal{M}_N\) spectral / Picard diagnostics, including the
//!   inhomogeneous map \(\Phi = K\Phi + S\) and modular kernel comparison.
//! - Curvature-cutoff Hadamard estimate near \(\Sigma_{\mathrm{ERGO}}\) remains an
//!   analytical EFT outlook only (not computed by this engine).
//!
//! Certified numerical claims refer to discrete operator diagnostics on explicit
//! metric profiles—not continuum field theorems.

pub mod algebra;
pub mod dynamics;
pub mod energy;

#[cfg(test)]
mod tests {
    mod monodromy_test;
}
