//! Christoffel symbols of the second kind Γ^λ_{μν}.
//!
//! Γ^λ_{μν} = (1/2) g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν}).

use super::DIM;

/// Dense storage Γ[λ][μ][ν] for a 4D manifold.
pub type Christoffel = [[[f64; DIM]; DIM]; DIM];

/// Zero Christoffel array (scaffold; fill from finite differences or symbolic import).
pub fn zero_christoffel() -> Christoffel {
    [[[0.0; DIM]; DIM]; DIM]
}

/// Symmetrize lower indices (Γ^λ_{μν} = Γ^λ_{νμ} for torsion-free Levi-Civita).
pub fn enforce_torsion_free(gamma: &mut Christoffel) {
    for lambda in 0..DIM {
        for mu in 0..DIM {
            for nu in (mu + 1)..DIM {
                let avg = 0.5 * (gamma[lambda][mu][nu] + gamma[lambda][nu][mu]);
                gamma[lambda][mu][nu] = avg;
                gamma[lambda][nu][mu] = avg;
            }
        }
    }
}
