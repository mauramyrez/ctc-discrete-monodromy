#!/usr/bin/env python3
"""
Optional manuscript-figure helpers aligned with the Rust reference profile.

These Matplotlib scripts are documentation aids for the SoftwareX article; they are
not part of the Rust acceptance gate (`cargo test`). Regenerate vector PDFs under
docs/figures/ when illustration assets need refreshing:

  docs/figures/lightcones.pdf     — light-cone tipping in the (t, φ) subspace
  docs/figures/geodesic_loop.pdf  — RK4 geodesic orbit γ(τ) at r_loop = 1.5 r0
  docs/figures/ergosurface.pdf    — g_tt(r) and e^{2Φ} vs r²ω² with Σ_ERGO

Geometry (signature −+++, equatorial θ = π/2):
  ds² = −e^{2Φ} dt² + dr²/(1−b/r) + r² (dφ − ω dt)²

Ergoregion / frame-dragging criterion: e^{2Φ} < r² ω² (g_tt > 0; ergosurface Σ_ERGO).
This is an ergoregion diagnostic of extreme frame dragging, not a chronology-horizon claim.

Reference parameters match ExplicitProfileParams::prd_reference():
  r0=1, γ=0.5, α=0.1, ω0=1.2, β=2.0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "figures"

# --- Match Rust ExplicitProfileParams::prd_reference() ---------------------------
R0 = 1.0
GAMMA = 0.5
ALPHA = 0.1
OMEGA0 = 1.2
BETA = 2.0
R_LOOP = 1.5 * R0
DTAU = 0.01
N_STEPS = 200
FD_EPS = 1e-5


def b_of_r(r: np.ndarray | float) -> np.ndarray | float:
    return R0 * (R0 / np.asarray(r, dtype=float)) ** GAMMA


def phi_of_r(r: np.ndarray | float) -> np.ndarray | float:
    return -ALPHA * R0 / np.asarray(r, dtype=float)


def omega_of_r(r: np.ndarray | float) -> np.ndarray | float:
    rr = np.asarray(r, dtype=float)
    return OMEGA0 * np.exp(-BETA * (rr - R0) / R0)


def e2phi(r: float) -> float:
    return float(np.exp(2.0 * phi_of_r(r)))


def is_ergoregion(r: float) -> bool:
    return e2phi(r) < (r**2) * float(omega_of_r(r)) ** 2


def ergosurface_radius() -> float:
    """Outer root of e^{2Φ} = r² ω² for r > r0 (bracketed bisection)."""
    lo = R0 * (1.0 + 1e-6)
    hi = R0 * 20.0
    if not is_ergoregion(lo) or is_ergoregion(hi):
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if is_ergoregion(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def null_cone_slopes(r: float) -> tuple[float, float]:
    omega = float(omega_of_r(r))
    e_phi = float(np.exp(phi_of_r(r)))
    return omega - e_phi / r, omega + e_phi / r


def draw_light_cone(
    ax,
    t0: float,
    phi0: float,
    r: float,
    half_width: float = 0.35,
    color: str = "#1a1a1a",
    alpha: float = 0.55,
) -> None:
    m_m, m_p = null_cone_slopes(r)
    dt = half_width
    ax.plot(
        [t0, t0 + dt],
        [phi0, phi0 + m_p * dt],
        color=color,
        lw=1.15,
        alpha=alpha,
        solid_capstyle="round",
    )
    ax.plot(
        [t0, t0 + dt],
        [phi0, phi0 + m_m * dt],
        color=color,
        lw=1.15,
        alpha=alpha,
        solid_capstyle="round",
    )
    verts = np.array(
        [
            [t0, phi0],
            [t0 + dt, phi0 + m_p * dt],
            [t0 + dt, phi0 + m_m * dt],
        ]
    )
    ax.add_patch(
        Polygon(verts, closed=True, facecolor=color, edgecolor="none", alpha=0.12)
    )


def plot_lightcones(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.85), constrained_layout=True)

    radii = np.array([3.5, 2.5, 1.8, 1.5, 1.2, 1.05])
    t_centers = np.linspace(0.4, 5.2, len(radii))
    phi_base = 1.6

    for t0, r in zip(t_centers, radii):
        ergo = is_ergoregion(float(r))
        color = "#b00020" if ergo else "#1565c0"
        draw_light_cone(ax, t0, phi_base, float(r), half_width=0.42, color=color, alpha=0.85)
        ax.text(
            t0,
            phi_base - 0.55,
            rf"$r={r:.2f}$",
            ha="center",
            va="top",
            fontsize=7,
            color="#333",
        )
        if ergo:
            ax.annotate(
                "",
                xy=(t0 - 0.28, phi_base),
                xytext=(t0, phi_base),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#b00020",
                    lw=1.0,
                    mutation_scale=8,
                ),
            )
            ax.text(
                t0 - 0.32,
                phi_base + 0.18,
                r"$-\partial_t$",
                fontsize=6.5,
                color="#b00020",
                ha="right",
            )

    ax.axhline(phi_base, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
    ax.text(
        0.02,
        0.98,
        r"Ergoregion: $e^{2\Phi}<r^{2}\omega^{2}$ ($g_{tt}>0$)"
        "\n"
        r"blue: outside ergoregion"
        "\n"
        r"red: inside ergoregion ($\partial_t$ spacelike)"
        "\n"
        rf"$\alpha={ALPHA},\ \omega_0={OMEGA0},\ \beta={BETA}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\varphi$")
    ax.set_title(r"Light-cone tipping ($\theta=\pi/2$, PRD profile)", fontsize=9)
    ax.set_xlim(-0.2, 5.9)
    ax.set_ylim(0.6, 3.35)
    ax.tick_params(labelsize=7)

    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def metric_components(r: float, theta: float = np.pi / 2) -> np.ndarray:
    """Numeric g_μν matching Rust metric_at (constant Φ,b,ω at radius r)."""
    phi = float(phi_of_r(r))
    b = float(b_of_r(r))
    omega = float(omega_of_r(r))
    s2 = np.sin(theta) ** 2
    r2 = r * r
    r2s2 = r2 * s2
    e2 = np.exp(2.0 * phi)
    g = np.zeros((4, 4))
    g[0, 0] = -e2 + r2s2 * omega * omega
    g[0, 3] = g[3, 0] = -r2s2 * omega
    g[1, 1] = 1.0 / max(1.0 - b / r, 1e-12)
    g[2, 2] = r2
    g[3, 3] = r2s2
    return g


def invert_4x4(g: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(g)
    except np.linalg.LinAlgError:
        return np.diag([-1.0, 1.0, 1.0, 1.0])


def christoffel_fd(x: np.ndarray, eps: float = FD_EPS) -> np.ndarray:
    """Central-FD Christoffels at x, matching Rust christoffel_fd spirit."""
    gamma = np.zeros((4, 4, 4))
    # Metric depends on r, θ only for this stationary axisymmetric ansatz
    r = max(float(x[1]), 1e-6)
    theta = float(np.clip(x[2], 1e-6, np.pi - 1e-6))
    g = metric_components(r, theta)
    ginv = invert_4x4(g)

    dg = np.zeros((4, 4, 4))
    for mu in (1, 2):  # ∂_r, ∂_θ
        xp = x.copy()
        xm = x.copy()
        xp[mu] += eps
        xm[mu] -= eps
        rp = max(float(xp[1]), 1e-6)
        rm = max(float(xm[1]), 1e-6)
        thp = float(np.clip(xp[2], 1e-6, np.pi - 1e-6))
        thm = float(np.clip(xm[2], 1e-6, np.pi - 1e-6))
        dg[mu] = (metric_components(rp, thp) - metric_components(rm, thm)) / (2.0 * eps)

    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = 0.0
                for sig in range(4):
                    acc += ginv[lam, sig] * (
                        dg[mu, nu, sig] + dg[nu, mu, sig] - dg[sig, mu, nu]
                    )
                gamma[lam, mu, nu] = 0.5 * acc
    return gamma


def geodesic_rhs(state: np.ndarray) -> np.ndarray:
    x = state[:4]
    u = state[4:]
    # Throat clamp as in Rust geodesic provider
    x = x.copy()
    x[1] = max(x[1], 1.05 * float(b_of_r(max(x[1], R0))) + 0.05)
    x[2] = float(np.clip(x[2], 1e-6, np.pi - 1e-6))
    gamma = christoffel_fd(x)
    du = np.zeros(4)
    for lam in range(4):
        acc = 0.0
        for mu in range(4):
            for nu in range(4):
                acc -= gamma[lam, mu, nu] * u[mu] * u[nu]
        du[lam] = acc
    return np.concatenate([u, du])


def rk4_step(state: np.ndarray, dtau: float) -> np.ndarray:
    k1 = geodesic_rhs(state)
    k2 = geodesic_rhs(state + 0.5 * dtau * k1)
    k3 = geodesic_rhs(state + 0.5 * dtau * k2)
    k4 = geodesic_rhs(state + dtau * k3)
    return state + (dtau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def corotating_initial_state(r_loop: float = R_LOOP) -> np.ndarray:
    """u ∝ (1, 0, 0, ω), normalized to g(u,u) = −1 (Rust corotating_initial_state)."""
    g = metric_components(r_loop)
    omega = float(omega_of_r(r_loop))
    u = np.array([1.0, 0.0, 0.0, omega])
    norm = float(u @ g @ u)
    assert norm < 0.0, f"co-rotating tangent must be timelike: g(u,u)={norm}"
    u /= np.sqrt(-norm)
    x = np.array([0.0, r_loop, np.pi / 2, 0.0])
    return np.concatenate([x, u])


def integrate_rk4_geodesic(
    r_loop: float = R_LOOP,
    dtau: float = DTAU,
    n_steps: int = N_STEPS,
) -> np.ndarray:
    """Return trajectory array of shape (n_steps+1, 4) for coordinates x^μ."""
    state = corotating_initial_state(r_loop)
    traj = [state[:4].copy()]
    for _ in range(n_steps):
        state = rk4_step(state, dtau)
        traj.append(state[:4].copy())
    return np.asarray(traj)


def plot_geodesic_loop(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)

    traj = integrate_rk4_geodesic()
    r = traj[:, 1]
    phi = traj[:, 3]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    phi_circle = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        R_LOOP * np.cos(phi_circle),
        R_LOOP * np.sin(phi_circle),
        color="#888888",
        lw=1.0,
        ls="--",
        label=rf"$r_{{\mathrm{{loop}}}}={R_LOOP}$",
        zorder=1,
    )

    r_ch = ergosurface_radius()
    if np.isfinite(r_ch):
        ax.plot(
            r_ch * np.cos(phi_circle),
            r_ch * np.sin(phi_circle),
            color="#e65100",
            lw=1.1,
            ls=":",
            label=rf"$\Sigma_{{\mathrm{{ERGO}}}}$ ($r\approx{r_ch:.2f}$)",
            zorder=2,
        )

    ax.plot(
        R0 * np.cos(phi_circle),
        R0 * np.sin(phi_circle),
        color="#424242",
        lw=1.4,
        label=rf"throat $r=r_0={R0}$",
        zorder=2,
    )

    ax.plot(x, y, color="#6a1b9a", lw=2.0, label=r"RK4 $\gamma(\tau)$", zorder=3)
    k = max(len(x) // 8, 1)
    ax.annotate(
        "",
        xy=(x[k + 3], y[k + 3]),
        xytext=(x[k], y[k]),
        arrowprops=dict(arrowstyle="-|>", color="#6a1b9a", lw=1.2, mutation_scale=10),
        zorder=4,
    )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x = r\cos\varphi$")
    ax.set_ylabel(r"$y = r\sin\varphi$")
    ax.set_title(
        rf"RK4 geodesic ($\Delta\tau={DTAU}$, ${N_STEPS}$ steps)",
        fontsize=9,
    )
    ax.legend(
        loc="upper right",
        fontsize=6.2,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )
    ax.tick_params(labelsize=7)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.5, zorder=0)

    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def g_tt_eq(r: np.ndarray | float) -> np.ndarray | float:
    """Equatorial g_tt = -e^{2Φ} + r² ω²."""
    rr = np.asarray(r, dtype=float)
    return -np.exp(2.0 * phi_of_r(rr)) + (rr**2) * omega_of_r(rr) ** 2


def plot_ergosurface(path: Path) -> None:
    """Two-panel diagnostic: competing scalars and g_tt(r) with Σ_ERGO markers."""
    r_ergo = ergosurface_radius()
    r = np.linspace(R0 * 1.001, 3.0, 600)
    e2 = np.exp(2.0 * phi_of_r(r))
    r2w2 = (r**2) * omega_of_r(r) ** 2
    gtt = g_tt_eq(r)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(3.4, 3.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )

    # --- Top: e^{2Φ} vs r²ω² ---
    ax0.plot(r, e2, color="#1565c0", lw=1.6, label=r"$e^{2\Phi(r)}$")
    ax0.plot(r, r2w2, color="#b00020", lw=1.6, label=r"$r^{2}\omega(r)^{2}$")
    ax0.fill_between(
        r,
        e2,
        r2w2,
        where=(r2w2 > e2),
        color="#b00020",
        alpha=0.12,
        interpolate=True,
        label=r"ergoregion ($g_{tt}>0$)",
    )
    ax0.axvline(R0, color="#333333", lw=0.9, ls=":", label=r"$r_0$")
    ax0.axvline(r_ergo, color="#000000", lw=1.1, ls="--", label=r"$\Sigma_{\mathrm{ERGO}}$")
    ax0.axvline(R_LOOP, color="#2e7d32", lw=1.1, ls="-.", label=r"$r_{\mathrm{loop}}$")
    ax0.set_ylabel(r"scalar amplitude")
    ax0.set_ylim(0.0, 1.55)
    ax0.legend(loc="upper right", fontsize=5.8, frameon=True, fancybox=False, edgecolor="#cccccc")
    ax0.tick_params(labelsize=7)
    ax0.set_title(r"Ergosurface diagnostic (equatorial, reference profile)", fontsize=9)

    # --- Bottom: g_tt(r) ---
    ax1.axhline(0.0, color="#888888", lw=0.8, zorder=0)
    ax1.fill_between(
        r,
        0.0,
        gtt,
        where=(gtt > 0),
        color="#b00020",
        alpha=0.18,
        interpolate=True,
    )
    ax1.fill_between(
        r,
        0.0,
        gtt,
        where=(gtt < 0),
        color="#1565c0",
        alpha=0.10,
        interpolate=True,
    )
    ax1.plot(r, gtt, color="#111111", lw=1.7, label=r"$g_{tt}(r)$")
    ax1.axvline(R0, color="#333333", lw=0.9, ls=":")
    ax1.axvline(r_ergo, color="#000000", lw=1.1, ls="--")
    ax1.axvline(R_LOOP, color="#2e7d32", lw=1.1, ls="-.")
    ax1.plot([r_ergo], [0.0], "o", color="#000000", ms=4.5, zorder=5)
    ax1.annotate(
        rf"$r_{{\mathrm{{ERGO}}}}\approx{r_ergo:.3f}\,r_0$",
        xy=(r_ergo, 0.0),
        xytext=(r_ergo + 0.35, 0.28),
        fontsize=6.5,
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8),
    )
    ax1.annotate(
        r"$r_{\mathrm{loop}}$ (C3)",
        xy=(R_LOOP, float(g_tt_eq(R_LOOP))),
        xytext=(R_LOOP + 0.25, float(g_tt_eq(R_LOOP)) - 0.35),
        fontsize=6.5,
        color="#2e7d32",
        arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=0.8),
    )
    ax1.set_xlabel(r"$r/r_0$")
    ax1.set_ylabel(r"$g_{tt}$")
    ax1.set_xlim(R0, 3.0)
    ax1.set_ylim(-1.05, 0.75)
    ax1.tick_params(labelsize=7)
    ax1.text(
        0.02,
        0.92,
        r"red: ergoregion"
        "\n"
        r"blue: exterior",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {path} (r_ERGO={r_ergo:.6f})")


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plot_lightcones(FIGDIR / "lightcones.pdf")
    plot_geodesic_loop(FIGDIR / "geodesic_loop.pdf")
    plot_ergosurface(FIGDIR / "ergosurface.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
