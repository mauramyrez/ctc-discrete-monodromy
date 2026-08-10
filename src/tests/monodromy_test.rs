//! Regression tests for discrete monodromy operator \(\mathcal{M}_N\).
//!
//! Verifies spectral / Picard software diagnostics of the row-sum renormalized
//! discrete kernel \(K\) on \(\mathbb{R}^N\) with `field.len() == n_nodes` enforced.
//! Homogeneous Picard (\(S=0\)) converges to \(\Phi^*\approx 0\); inhomogeneous
//! forcing yields a nontrivial discrete response vector.
//! Path-3 tests probe the **unnormalized** proxy \(\widetilde{K}\).

use crate::dynamics::monodromy::{
    alternate_initial_data_picard_test, banach_verification, continuum_limit_traces,
    default_forcing_vector, default_monodromy_operator, find_fixed_point,
    find_fixed_point_inhomogeneous, format_unnormalized_spectral_sweep,
    homogeneous_picard_baseline_test, KernelKind, path3_default_m2_grid, path3_default_n_grid,
    run_path3_extended_campaign, sweep_unnormalized_spectral_radius, PeriodicField, RenormPolicy,
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

/// Legacy Path-3 grid (fast). Run with: `cargo test path3_unnormalized -- --nocapture`
#[test]
fn path3_unnormalized_mass_and_n_sweep() {
    let samples =
        sweep_unnormalized_spectral_radius(&path3_default_n_grid(), &path3_default_m2_grid());
    println!("\n{}", format_unnormalized_spectral_sweep(&samples));
    assert_eq!(samples.len(), 12);
    assert_eq!(
        default_monodromy_operator().renorm,
        RenormPolicy::RowSumBenchmark
    );
}

/// Extended Path-3: \(N\le 1024\), \(m^2\to 0\), \(\nu=0\); writes `docs/path3_continuum_report.md`.
///
/// Run with: `cargo test path3_extended -- --nocapture`
#[test]
fn path3_extended_highn_ultralow_mass_report() {
    let report_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("path3_continuum_report.md");
    let (yukawa, expo, path) =
        run_path3_extended_campaign(&report_path).expect("Path-3 report write");
    assert_eq!(yukawa.len(), 36);
    assert_eq!(expo.len(), 36);
    assert!(path.is_file());

    println!("\n=== ExponentialDecay ν=0 (Coulomb/log) ===");
    println!("{}", format_unnormalized_spectral_sweep(&expo));
    println!("\n=== YukawaProxy ν=0 ===");
    println!("{}", format_unnormalized_spectral_sweep(&yukawa));

    for t in continuum_limit_traces(&expo)
        .into_iter()
        .chain(continuum_limit_traces(&yukawa))
    {
        println!(
            "N=1024 limit | {} | m²={:.1e} | ρ(1024)={} | {}",
            match t.kernel_kind {
                KernelKind::YukawaProxy => "Yukawa",
                KernelKind::ExponentialDecay => "ExpDecay",
            },
            t.m_squared,
            t.rho_at_1024
                .map(|r| format!("{r:.6e}"))
                .unwrap_or_else(|| "—".into()),
            t.verdict
        );
        assert!(t.rho_at_1024.is_some(), "missing ρ(1024)");
    }

    // Massless exponential ν=0: ρ → (N-1)Δτ → 2π√g_φφ ≳ 1 (true Coulomb rupture).
    assert!(expo
        .iter()
        .filter(|s| s.m_squared == 0.0)
        .all(|s| s.ruptured()));

    // Yukawa ∝ m_* collapses as m→0 at ν=0.
    assert!(yukawa
        .iter()
        .filter(|s| s.m_squared == 0.0)
        .all(|s| s.contractive && s.rho < 1e-12));

    println!("Report: {}", path.display());
}
