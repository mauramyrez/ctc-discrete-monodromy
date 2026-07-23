//! Tensor calculus: metric definitions, Christoffel symbols, curvature scalars.

pub mod christoffel;
pub mod invariants;
pub mod metric;
pub mod metric_numeric;
pub mod metric_profiles;

pub use metric::DIM;
