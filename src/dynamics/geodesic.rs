//! Geodesic equations: d²x^λ/dτ² + Γ^λ_{μν} u^μ u^ν = 0.
//!
//! Implements 4th-order Runge–Kutta integration with metric-dependent Christoffel
//! evaluation for the Morris–Thorne frame-dragging background.

use crate::algebra::christoffel::Christoffel;
use crate::algebra::metric_numeric::{christoffel_fd, MorrisThorneParams};
use crate::algebra::DIM;

/// Phase-space state: (x^μ, u^μ) with u^μ = dx^μ/dτ.
#[derive(Clone, Debug, PartialEq)]
pub struct GeodesicState {
    pub x: [f64; DIM],
    pub u: [f64; DIM],
}

/// Callable Christoffel provider at a spacetime point.
pub trait ChristoffelProvider {
    fn at(&self, x: &[f64; DIM]) -> Christoffel;
    fn as_mt(&self) -> Option<&MtChristoffel> {
        None
    }
}

/// Morris–Thorne Christoffel provider (uses r = x¹, θ = x²).
pub struct MtChristoffel {
    pub params_fn: Box<dyn Fn(f64) -> MorrisThorneParams + Send + Sync>,
}

impl ChristoffelProvider for MtChristoffel {
    fn at(&self, x: &[f64; DIM]) -> Christoffel {
        let r = x[1].max(1e-6);
        let theta = x[2].clamp(1e-6, std::f64::consts::PI - 1e-6);
        let xq = [x[0], r, theta, x[3]];
        christoffel_fd(&xq, &|rad| (self.params_fn)(rad), 1e-5)
    }

    fn as_mt(&self) -> Option<&MtChristoffel> {
        Some(self)
    }
}

/// Right-hand side of the first-order geodesic system (dx/dτ, du/dτ).
pub fn geodesic_rhs(gamma: &Christoffel, state: &GeodesicState) -> ([f64; DIM], [f64; DIM]) {
    let mut dx = [0.0; DIM];
    let mut du = [0.0; DIM];
    for lambda in 0..DIM {
        dx[lambda] = state.u[lambda];
        let mut accel = 0.0;
        for mu in 0..DIM {
            for nu in 0..DIM {
                accel -= gamma[lambda][mu][nu] * state.u[mu] * state.u[nu];
            }
        }
        du[lambda] = accel;
    }
    (dx, du)
}

fn add_state(a: &GeodesicState, dx: &[f64; DIM], du: &[f64; DIM], scale: f64) -> GeodesicState {
    let mut x = [0.0; DIM];
    let mut u = [0.0; DIM];
    for i in 0..DIM {
        x[i] = a.x[i] + scale * dx[i];
        u[i] = a.u[i] + scale * du[i];
    }
    GeodesicState { x, u }
}

/// Classical 4th-order Runge–Kutta step for the geodesic ODE.
pub fn rk4_step(
    provider: &dyn ChristoffelProvider,
    state: &GeodesicState,
    dtau: f64,
) -> GeodesicState {
    let g0 = provider.at(&state.x);
    let (k1x, k1u) = geodesic_rhs(&g0, state);

    let mut s2 = add_state(state, &k1x, &k1u, 0.5 * dtau);
    clamp_outside_throat(&mut s2, provider);
    let g2 = provider.at(&s2.x);
    let (k2x, k2u) = geodesic_rhs(&g2, &s2);

    let mut s3 = add_state(state, &k2x, &k2u, 0.5 * dtau);
    clamp_outside_throat(&mut s3, provider);
    let g3 = provider.at(&s3.x);
    let (k3x, k3u) = geodesic_rhs(&g3, &s3);

    let mut s4 = add_state(state, &k3x, &k3u, dtau);
    clamp_outside_throat(&mut s4, provider);
    let g4 = provider.at(&s4.x);
    let (k4x, k4u) = geodesic_rhs(&g4, &s4);

    let mut next = state.clone();
    for i in 0..DIM {
        next.x[i] += (dtau / 6.0) * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
        next.u[i] += (dtau / 6.0) * (k1u[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + k4u[i]);
    }
    clamp_outside_throat(&mut next, provider);
    next
}

/// Integrate geodesic from state over n_steps with step dtau.
/// Keeps r outside the throat (r > b + margin) to avoid metric singularities.
pub fn integrate_geodesic(
    provider: &dyn ChristoffelProvider,
    initial: &GeodesicState,
    dtau: f64,
    n_steps: usize,
) -> Vec<GeodesicState> {
    let mut traj = Vec::with_capacity(n_steps + 1);
    let mut state = initial.clone();
    traj.push(state.clone());
    for _ in 0..n_steps {
        state = rk4_step(provider, &state, dtau);
        clamp_outside_throat(&mut state, provider);
        traj.push(state.clone());
    }
    traj
}

/// Prevent r from entering the throat where g_rr diverges.
fn clamp_outside_throat(state: &mut GeodesicState, provider: &dyn ChristoffelProvider) {
    if let Some(mt) = provider.as_mt() {
        let p = (mt.params_fn)(state.x[1]);
        let r_min = p.b * 1.05 + 0.05;
        state.x[1] = state.x[1].max(r_min);
    }
    state.x[2] = state.x[2].clamp(0.05, std::f64::consts::PI - 0.05);
}

/// Co-rotating azimuthal loop: timelike tangent u^μ ∝ (1, 0, 0, ω(r)), normalized with g(u,u)=−1.
pub fn corotating_initial_state(r: f64, omega0: f64, b0: f64, phi0: f64) -> GeodesicState {
    let p = MorrisThorneParams::frame_dragging_example(r, b0, omega0, phi0);
    corotating_initial_state_from_params(r, &p)
}

/// Co-rotating initial state from explicit metric parameters at radius r.
pub fn corotating_initial_state_from_params(r: f64, p: &MorrisThorneParams) -> GeodesicState {
    let theta = std::f64::consts::FRAC_PI_2;
    let (g, _) = crate::algebra::metric_numeric::metric_at(r, theta, p);
    let omega = p.omega;
    let ut = 1.0;
    let uphi = omega;
    let norm = g[0][0] * ut * ut + 2.0 * g[0][3] * ut * uphi + g[3][3] * uphi * uphi;
    assert!(
        norm < 0.0,
        "co-rotating tangent must be timelike: g(ξ,ξ)={norm}"
    );
    let scale = (-1.0 / norm).sqrt();
    GeodesicState {
        x: [0.0, r, theta, 0.0],
        u: [ut * scale, 0.0, 0.0, uphi * scale],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::metric_profiles::ExplicitProfileParams;

    #[test]
    fn rk4_integrates_azimuthal_loop() {
        // Reference profile: (r0, γ, α, ω0, β) = (1, 0.5, 0.1, 1.2, 2.0)
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
        assert!(disp > 1e-6, "geodesic must propagate along co-rotating loop");
        assert!((last.x[1] - r_loop).abs() < 0.2);
    }
}
