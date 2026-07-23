//! Numerical validation of discrete Novikov consistency for \(\mathcal{M}_N\) (criterion C3).
//!
//! Verifies Banach–Picard convergence of the row-sum renormalized discrete
//! kernel \(K\) on \(\mathbb{R}^N\) with `field.len() == n_nodes` enforced.

use crate::dynamics::novikov::{
    banach_verification, default_monodromy_operator, find_fixed_point, nontrivial_fixed_point_test,
    novikov_fixed_point_test, PeriodicField,
};

#[test]
fn banach_contraction_gives_unique_fixed_point() {
    let m = default_monodromy_operator();
    assert!(banach_verification(&m));
    assert_eq!(m.n_nodes, 128);
    let init = PeriodicField::new((0..m.n_nodes).map(|i| i as f64 * 0.01).collect());
    assert_eq!(init.len(), m.n_nodes);
    let fp = find_fixed_point(&init, &m).unwrap();
    assert_eq!(fp.field.len(), m.n_nodes);
    assert!(fp.residual < 1e-10);
    assert!(fp.lipschitz < 1.0);
}

#[test]
fn novikov_sinusoidal_initial_data_converges() {
    let fp = novikov_fixed_point_test().unwrap();
    assert_eq!(fp.field.len(), 128);
    assert!(fp.residual < 1e-8);
    println!(
        "Novikov discrete M_N: converged in {} iterations, residual = {:.3e}, L = {:.4}",
        fp.iterations, fp.residual, fp.lipschitz
    );
}

#[test]
fn nontrivial_fixed_point_exists() {
    let fp = nontrivial_fixed_point_test().unwrap();
    assert_eq!(fp.field.len(), 64);
    assert!(fp.residual < 1e-8);
    let mean: f64 = fp.field.values.iter().sum::<f64>() / fp.field.len() as f64;
    assert!(mean.is_finite());
    println!("Non-trivial fixed-point mean = {mean:.6}");
}

#[test]
fn geodesic_ctc_loop_integrates() {
    use crate::algebra::metric_profiles::ExplicitProfileParams;
    use crate::dynamics::geodesic::{
        ctc_initial_state_from_params, integrate_geodesic, MtChristoffel,
    };
    let prof = ExplicitProfileParams::prd_reference();
    let r_loop = 1.5 * prof.r0;
    let provider = MtChristoffel {
        params_fn: Box::new(move |r| ExplicitProfileParams::prd_reference().at(r)),
    };
    let mt = prof.at(r_loop);
    let init = ctc_initial_state_from_params(r_loop, &mt);
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
}
