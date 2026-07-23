//! Stress-energy API scaffold for static energy-condition evaluation.
//!
//! Sample tensors support unit-test quadratic forms only. Manuscript NEC/WEC
//! contractions are SymPy background assets (criterion C2), not dynamically
//! evolved Rust curvature tensors.

pub mod conditions;
pub mod stress_energy;
