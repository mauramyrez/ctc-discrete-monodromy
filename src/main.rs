//! CTC Research Framework — driver for discrete monodromy and scaffold diagnostics.

use ctc_research_framework::algebra::invariants;
use ctc_research_framework::dynamics::{bvp, novikov};
use ctc_research_framework::energy::conditions;

use anyhow::Result;

fn main() -> Result<()> {
    println!("CTC Research Framework");
    println!("Discrete Sobolev monodromy contraction (criterion C3)");
    println!("Signature: (-+++); geometric units c = G = 1");
    println!();

    let report = invariants::sample_metric_report()?;
    println!("[algebra] {}", report);

    let fp = novikov::novikov_fixed_point_test()?;
    println!(
        "[dynamics] Discrete M_N Picard: {} iters, residual = {:.3e}, L = {:.4}",
        fp.iterations, fp.residual, fp.lipschitz
    );

    let smoke = bvp::novikov_fixed_point_smoke_test()?;
    println!("[dynamics] Smoke test: {}", smoke);

    let nec = conditions::null_energy_condition_sample()?;
    let wec = conditions::weak_energy_condition_sample()?;
    println!("[energy] API scaffold NEC sample ≥ 0: {}", nec);
    println!("[energy] API scaffold WEC sample ≥ 0: {}", wec);

    Ok(())
}
