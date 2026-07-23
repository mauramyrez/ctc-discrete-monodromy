//! Sample stress-energy tensors for the energy-condition API scaffold.
//!
//! Not assembled from Rust curvature; SymPy supplies manuscript \(G_{\mu\nu}/8\pi\).

use crate::algebra::DIM;
use nalgebra::Matrix4;

pub type Tensor4 = Matrix4<f64>;

/// Contract \(T_{\mu\nu} v^\mu v^\nu\).
pub fn quadratic_form(t: &Tensor4, v: &[f64; DIM]) -> f64 {
    let mut acc = 0.0;
    for mu in 0..DIM {
        for nu in 0..DIM {
            acc += t[(mu, nu)] * v[mu] * v[nu];
        }
    }
    acc
}

/// Sample exotic-matter–like \(T_{\mu\nu}\) with negative lab-frame energy density.
/// API-scaffold only (criterion C2 companion); not a dynamical Rust tensor.
pub fn sample_exotic_stress_energy() -> Tensor4 {
    let mut t = Tensor4::zeros();
    t[(0, 0)] = -0.1;
    t[(1, 1)] = 0.05;
    t[(2, 2)] = 0.05;
    t[(3, 3)] = 0.05;
    t
}

/// Sample ordinary dust-like \(T_{\mu\nu}\) with \(\rho > 0\) (scaffold).
pub fn sample_ordinary_stress_energy() -> Tensor4 {
    let mut t = Tensor4::zeros();
    t[(0, 0)] = 1.0;
    t
}
