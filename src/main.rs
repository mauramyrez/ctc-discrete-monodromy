//! Discrete operator diagnostics driver (spectral / Picard / inhomogeneous response).

use ctc_research_framework::algebra::invariants;
use ctc_research_framework::algebra::metric_profiles::ExplicitProfileParams;
use ctc_research_framework::dynamics::monodromy::{
    default_forcing_vector, default_monodromy_operator, find_fixed_point,
    find_fixed_point_inhomogeneous, KernelKind, PeriodicField,
};
use ctc_research_framework::dynamics::{bvp, monodromy};
use ctc_research_framework::energy::conditions;

use anyhow::Result;

fn main() -> Result<()> {
    println!("Discrete Operator Diagnostics Framework");
    println!("Row-normalized monodromy spectral / Picard / inhomogeneous baseline");
    println!("Signature: (-+++); geometric units c = G = 1");
    println!();

    let report = invariants::sample_metric_report()?;
    println!("[algebra] {}", report);

    let prof = ExplicitProfileParams::prd_reference();
    if let Some(r_ergo) = prof.ergosurface_radius() {
        let r_loop = 1.5 * prof.r0;
        let mt_loop = prof.at(r_loop);
        let g_tt_loop =
            (-(2.0 * mt_loop.phi).exp()) + r_loop * r_loop * mt_loop.omega * mt_loop.omega;
        println!(
            "[geometry] r_ERGO ≈ {:.6}, r_loop = {:.3}, g_tt(r_loop) ≈ {:.6e} (spacelike ∂_t iff > 0)",
            r_ergo, r_loop, g_tt_loop
        );
    }

    println!("[dynamics] M_N homogeneous Picard table (m^2=1, ν=1, r_loop=1.5 r0):");
    println!("N\trho(K)\tL_osc\tresidual\titers\t||Φ*||_2");
    for &n in &[32usize, 64, 128] {
        let mut m = default_monodromy_operator();
        m.n_nodes = n;
        let init = PeriodicField::new(
            (0..n)
                .map(|i| 1.0 + 0.1 * (i as f64) + 0.05 * (i as f64 * 0.2).sin())
                .collect(),
        );
        let rho = m.l2_operator_norm();
        let l_osc = m.lipschitz_bound();
        let fp = find_fixed_point(&init, &m)?;
        println!(
            "{}\t{:.6}\t{:.6}\t{:.3e}\t{}\t{:.3e}",
            n, rho, l_osc, fp.residual, fp.iterations, fp.field.l2_norm()
        );
    }

    let fp = monodromy::homogeneous_picard_baseline_test()?;
    println!(
        "[dynamics] Default N=128 homogeneous Picard: {} iters, residual = {:.3e}, L = {:.4}, ||Φ*||_2 = {:.3e}",
        fp.iterations, fp.residual, fp.lipschitz, fp.field.l2_norm()
    );

    let m = default_monodromy_operator();
    let source = default_forcing_vector(m.n_nodes);
    let init = PeriodicField::new(vec![0.0; m.n_nodes]);
    let inhomo = find_fixed_point_inhomogeneous(&init, &m, &source)?;
    println!(
        "[dynamics] Inhomogeneous Φ=KΦ+S (N=128): {} iters, residual = {:.3e}, ||Φ*||_2 = {:.3e}",
        inhomo.iterations, inhomo.residual, inhomo.field.l2_norm()
    );

    let mut yukawa = default_monodromy_operator();
    yukawa.n_nodes = 64;
    let mut expo = yukawa.clone();
    expo.kernel_kind = KernelKind::ExponentialDecay;
    println!(
        "[dynamics] Modular kernels @ N=64: rho(Yukawa)={:.6}, rho(ExpDecay)={:.6}",
        yukawa.spectral_radius(),
        expo.spectral_radius()
    );

    let smoke = bvp::homogeneous_picard_smoke_test()?;
    println!("[dynamics] Homogeneous smoke: {}", smoke);
    let smoke_i = bvp::inhomogeneous_picard_smoke_test()?;
    println!("[dynamics] Inhomogeneous smoke: {}", smoke_i);

    let nec = conditions::null_energy_condition_sample()?;
    let wec = conditions::weak_energy_condition_sample()?;
    println!("[energy] API scaffold NEC sample ≥ 0: {}", nec);
    println!("[energy] API scaffold WEC sample ≥ 0: {}", wec);

    Ok(())
}
