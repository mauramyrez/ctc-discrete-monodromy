//! CTC Research Framework library.
//!
//! Signature: (−+++). Geometric units \(c = G = 1\).
//!
//! Computational status mirrors the manuscript criteria (C1)–(C4):
//! - **(C1)** Explicit frame-dragging Morris–Thorne profiles and CTC locus (implemented).
//! - **(C2)** SymPy-exported equatorial Einstein / energy-condition assets (static; not Rust-dynamical).
//! - **(C3)** Discrete Yukawa-proxy monodromy \(\mathcal{M}_N\) Banach–Picard contraction (core Rust result).
//! - **(C4)** Curvature-cutoff Hadamard bound (analytic EFT outlook only; not computed here).

pub mod algebra;
pub mod dynamics;
pub mod energy;

#[cfg(test)]
mod tests {
    mod novikov_test;
}
