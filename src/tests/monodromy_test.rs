//! Regression tests for discrete monodromy operator \(\mathcal{M}_N\).
//!
//! Verifies spectral / Picard software diagnostics of the row-sum renormalized
//! discrete kernel \(K\) on \(\mathbb{R}^N\) with `field.len() == n_nodes` enforced.
//! Homogeneous Picard (\(S=0\)) converges to \(\Phi^*\approx 0\); inhomogeneous
//! forcing yields a nontrivial discrete response vector.

use crate::dynamics::monodromy::{
    alternate_initial_data_picard_test, banach_verification, default_forcing_vector,
    default_monodromy_operator, find_fixed_point, find_fixed_point_inhomogeneous,
    homogeneous_picard_baseline_test, PeriodicField,
};

#[test]
fn banach_contraction_gives_trivial_homogeneous_fixed_point() {
    let m = default_monodromy_operator();
    assert!(banach_verification(&m));
    assert_eq!(m.n_nodes, 128);
    let init = PeriodicField::new((0..m.n_nodes).map(|i| i as f64 * 0.01).collect());
    assert_eq!(init.len(), m.n_nodes);
    let fp = find_fixed_point(&init, &m).unwrap();
    assert_eq!(fp.field.len(), m.n_nodes);
    assert!(fp.residual < 1e-10);
    assert!(fp.lipschitz < 1.0);
    assert!(fp.field.l2_norm() < 1e-6);
}

#[test]
fn homogeneous_picard_sinusoidal_initial_data_converges() {
    let fp = homogeneous_picard_baseline_test().unwrap();
    assert_eq!(fp.field.len(), 128);
    assert!(fp.residual < 1e-8);
    assert!(fp.field.l2_norm() < 1e-6);
    println!(
        "Homogeneous M_N Picard: converged in {} iterations, residual = {:.3e}, L = {:.4}, ||Φ*||_2 = {:.3e}",
        fp.iterations, fp.residual, fp.lipschitz, fp.field.l2_norm()
    );
}

#[test]
fn alternate_initial_data_also_converges_to_zero() {
    let fp = alternate_initial_data_picard_test().unwrap();
    assert_eq!(fp.field.len(), 64);
    assert!(fp.residual < 1e-8);
    assert!(fp.field.l2_norm() < 1e-6);
    let mean: f64 = fp.field.values.iter().sum::<f64>() / fp.field.len() as f64;
    assert!(mean.abs() < 1e-6);
    println!("Alternate-initial Picard mean = {mean:.6e} (expect ~0)");
}

#[test]
fn geodesic_corotating_loop_integrates() {
    use crate::algebra::metric_profiles::ExplicitProfileParams;
    use crate::dynamics::geodesic::{
        corotating_initial_state_from_params, integrate_geodesic, MtChristoffel,
    };
    let prof = ExplicitProfileParams::prd_reference();
    let r_loop = 1.5 * prof.r0;
    let provider = MtChristoffel {
        params_fn: Box::new(move |r| ExplicitProfileParams::prd_reference().at(r)),
    };
    let mt = prof.at(r_loop);
    let init = corotating_initial_state_from_params(r_loop, &mt);
    let traj = integrate_geodesic(&provider, &init, 0.01, 200);
    let last = traj.last().unwrap();
    let disp = (last.x[0] - init.x[0]).abs() + (last.x[3] - init.x[3]).abs();
    assert!(disp > 1e-6);
    assert!((last.x[1] - r_loop).abs() < 0.2);
}

#[test]
fn discrete_monodromy_spectral_radius_below_unity() {
    let m = default_monodromy_operator();
    assert!(m.l2_operator_norm() < 1.0);
    assert!(m.is_contraction());
    // Inhomogeneous response sanity: nonzero S ⇒ nontrivial Φ*.
    let source = default_forcing_vector(m.n_nodes);
    let init = PeriodicField::new(vec![0.0; m.n_nodes]);
    let fp = find_fixed_point_inhomogeneous(&init, &m, &source).unwrap();
    assert!(fp.residual < 1e-10);
    assert!(fp.field.l2_norm() > 1e-3);
}
