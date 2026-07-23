//! Explicit closed-form Morris–Thorne metric profiles (criterion C1).
//!
//! ```text
//! b(r)   = r_0 (r_0/r)^γ
//! Φ(r)   = −α r_0/r
//! ω(r)   = ω_0 exp(−β(r−r_0)/r_0)
//! ```

use super::metric_numeric::{ergoregion, MorrisThorneParams};

/// Admissibility parameters for the explicit profile class.
#[derive(Clone, Debug)]
pub struct ExplicitProfileParams {
    pub r0: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub omega0: f64,
    pub beta: f64,
}

impl ExplicitProfileParams {
    /// Reference profile: ergoregion opens outside throat, flare-out satisfied.
    pub fn prd_reference() -> Self {
        Self {
            r0: 1.0,
            gamma: 0.5,
            alpha: 0.1,
            omega0: 1.2,
            beta: 2.0,
        }
    }

    /// Flare-out at throat: b'(r_0) = −γ < 1 requires γ > 0.
    pub fn flare_out_satisfied(&self) -> bool {
        self.gamma > 0.0
    }

    /// Ergoregion opens at throat: e^{−2α} < r_0² ω_0².
    pub fn ergoregion_at_throat(&self) -> bool {
        (-2.0 * self.alpha).exp() < self.r0 * self.r0 * self.omega0 * self.omega0
    }

    /// Outer ergosurface radius \(r_{\mathrm{ERGO}}>r_0\) (first exit from
    /// \(e^{2\Phi}<r^{2}\omega^{2}\) when scanning outward from the throat).
    pub fn ergosurface_radius(&self) -> Option<f64> {
        let r0 = self.r0;
        let mut lo = r0 * (1.0 + 1e-6);
        let mut hi = r0 * 20.0;
        if !self.ergoregion_criterion(lo) {
            return None;
        }
        // Require an exterior point outside the ergoregion so the root is bracketed.
        if self.ergoregion_criterion(hi) {
            return None;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if self.ergoregion_criterion(mid) {
                lo = mid; // still inside ergoregion: raise lower bracket
            } else {
                hi = mid; // outside: lower upper bracket
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// e^{2Φ(r)} < r² ω(r)² on equatorial plane.
    pub fn ergoregion_criterion(&self, r: f64) -> bool {
        let p = self.at(r);
        ergoregion(r, &p)
    }

    /// Evaluate MorrisThorneParams at radius r.
    pub fn at(&self, r: f64) -> MorrisThorneParams {
        let r0 = self.r0;
        let b = r0 * (r0 / r).powf(self.gamma);
        let b_prime = -self.gamma * b / r;
        let phi = -self.alpha * r0 / r;
        let phi_prime = self.alpha * r0 / (r * r);
        let omega = self.omega0 * (-self.beta * (r - r0) / r0).exp();
        let omega_prime = -self.beta / r0 * omega;
        MorrisThorneParams {
            phi,
            b,
            omega,
            phi_prime,
            b_prime,
            omega_prime,
        }
    }

    /// Upper edge of compact exotic-matter domain: r_ERGO + δ.
    pub fn exotic_domain_upper(&self, delta: f64) -> Option<f64> {
        self.ergosurface_radius().map(|r_ergo| r_ergo + delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flare_out_and_ergoregion_admissibility() {
        let p = ExplicitProfileParams::prd_reference();
        assert!(p.flare_out_satisfied());
        assert!(p.ergoregion_at_throat());
        let r_ergo = p
            .ergosurface_radius()
            .expect("ergoregion should open outside throat");
        assert!(r_ergo > p.r0);
        assert!(p.ergoregion_criterion(1.05 * p.r0));
        assert!(!p.ergoregion_criterion(1.5 * p.r0));
        assert!(r_ergo > 1.05 * p.r0 && r_ergo < 1.5 * p.r0);
        let mt = p.at(r_ergo);
        let residual = (2.0 * mt.phi).exp() - r_ergo * r_ergo * mt.omega * mt.omega;
        assert!(
            residual.abs() < 5e-4,
            "ergosurface residual {residual} at r={r_ergo}"
        );
    }

    #[test]
    fn throat_shape_function_equals_r0() {
        let p = ExplicitProfileParams::prd_reference();
        let mt = p.at(p.r0);
        assert!((mt.b - p.r0).abs() < 1e-12);
        assert!((mt.b_prime + p.gamma).abs() < 1e-12);
    }
}
