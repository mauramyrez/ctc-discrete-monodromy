"""Shared symbolic tensor calculus for the rotating CTC Morris–Thorne metric."""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, Rational, collect, factor, simplify, symbols

COORDS = symbols("t r theta varphi", real=True)
T, R, THETA, PHI = COORDS

# Explicit metric profile parameters (GAP 1 closure)
R0, GAMMA, ALPHA, OMEGA0, BETA = symbols(
    "r_0 gamma alpha omega_0 beta", positive=True, real=True
)
DELTA = symbols("delta", positive=True, real=True)

# LaTeX labels for metric components (equatorial θ = π/2)
G_LABELS: dict[tuple[int, int], str] = {
    (0, 0): r"G_{tt}",
    (0, 3): r"G_{t\varphi}",
    (3, 0): r"G_{\varphi t}",
    (1, 1): r"G_{rr}",
    (2, 2): r"G_{\theta\theta}",
    (3, 3): r"G_{\varphi\varphi}",
}


def metric_functions():
    """Return Φ(r), b(r), ω(r) as SymPy functions."""
    Phi = sp.Function("Phi")(R)
    b = sp.Function("b")(R)
    omega = sp.Function("omega")(R)
    return Phi, b, omega


def explicit_metric_profiles() -> dict:
    """
    Closed-form Morris–Thorne profiles (PRD admissibility class).

    b(r)   = r_0 (r_0/r)^γ           flare-out: b'(r_0) = −γ < 1 for γ > 0
    Φ(r)   = −α r_0/r                 finite redshift at infinity
    ω(r)   = ω_0 exp(−β(r−r_0)/r_0)  localized frame dragging
    """
    Phi = -ALPHA * R0 / R
    b = R0 * (R0 / R) ** GAMMA
    omega = OMEGA0 * sp.exp(-BETA * (R - R0) / R0)
    return {"Phi": Phi, "b": b, "omega": omega}


def explicit_profile_derivatives() -> dict:
    """First and second radial derivatives of the explicit profiles."""
    prof = explicit_metric_profiles()
    out: dict = {}
    for name, func in prof.items():
        out[f"{name}_prime"] = sp.diff(func, R)
        out[f"{name}_double_prime"] = sp.diff(func, R, 2)
    return out


def substitute_explicit_profiles(expr: sp.Expr) -> sp.Expr:
    """Replace Φ(r), b(r), ω(r) function symbols with explicit profiles."""
    Phi, b, omega = metric_functions()
    subs = explicit_metric_profiles()
    return simplify_component(expr.subs({Phi: subs["Phi"], b: subs["b"], omega: subs["omega"]}))


def throat_flare_out_condition() -> sp.Expr:
    """b'(r_0) = −γ; flare-out requires b'(r_0) < 1 ⟺ γ > 0."""
    b = explicit_metric_profiles()["b"]
    return sp.diff(b, R).subs(R, R0)


def ctc_criterion_explicit() -> sp.Expr:
    """Equatorial CTC inequality: e^{2Φ} − r²ω² < 0."""
    prof = explicit_metric_profiles()
    return sp.exp(2 * prof["Phi"]) - R**2 * prof["omega"] ** 2


def ctc_onset_at_throat() -> sp.Expr:
    """Evaluate CTC criterion at r = r_0."""
    return ctc_criterion_explicit().subs(R, R0)


def energy_violation_domain_upper() -> sp.Expr:
    """Upper edge r_{CTC} + δ of compact exotic-matter domain."""
    r_ctc = symbols("r_CTC", positive=True, real=True)
    return r_ctc + DELTA


def parameter_admissibility_bounds() -> list[tuple[str, sp.Basic]]:
    """
    Analytical parameter bounds for (C1)–(C2).

    Returns list of (description, condition) pairs for LaTeX export.
    """
    prof = explicit_metric_profiles()
    b_prime_throat = throat_flare_out_condition()
    ctc_throat = ctc_onset_at_throat()
    return [
        (r"Flare-out at throat ($b'(r_0) = -\gamma < 1$)", sp.Gt(GAMMA, 0)),
        (
            r"CTC opens at throat ($e^{-2\alpha} < r_0^2 \omega_0^2$)",
            sp.Lt(sp.exp(-2 * ALPHA), R0**2 * OMEGA0**2),
        ),
        (r"Finite redshift at infinity ($\alpha \ge 0$)", sp.Ge(ALPHA, 0)),
        (r"Localized dragging ($\beta > 0$)", sp.Gt(BETA, 0)),
        (
            r"Exotic matter domain $[r_0,\, r_{\mathrm{CTC}}+\delta]$",
            sp.And(sp.Gt(DELTA, 0), sp.Gt(GAMMA, 0)),
        ),
    ]


def build_metric(equatorial: bool = False) -> tuple[Matrix, tuple, dict]:
    """
    Construct g_{μν} for the modified Morris–Thorne metric with frame dragging.

    If equatorial=True, set θ = π/2 (sin θ = 1) for the CTC azimuthal loop.
    """
    Phi, b, omega = metric_functions()
    e2Phi = sp.exp(2 * Phi)
    r2 = R**2
    sin_th = sp.Integer(1) if equatorial else sp.sin(THETA)
    r2s2 = r2 * sin_th**2

    g_tt = -e2Phi + r2s2 * omega**2
    g_rr = 1 / (1 - b / R)
    g_thth = r2
    g_phph = r2s2
    g_tphi = -r2s2 * omega

    g = Matrix(
        [
            [g_tt, 0, 0, g_tphi],
            [0, g_rr, 0, 0],
            [0, 0, g_thth, 0],
            [g_tphi, 0, 0, g_phph],
        ]
    )
    coords = COORDS
    funcs = {"Phi": Phi, "b": b, "omega": omega}
    return g, coords, funcs


def simplify_component(expr: sp.Expr) -> sp.Expr:
    """Aggressive algebraic simplification for export."""
    Phi, b, omega = metric_functions()
    expr = simplify(expr)
    expr = factor(expr)
    expr = collect(expr, [sp.exp(2 * Phi), sp.exp(-2 * Phi)])
    return simplify(expr)


def christoffel_second_kind(g: Matrix, coords: tuple) -> tuple[list, Matrix]:
    """Γ^λ_{μν} and g^{μν}."""
    n = g.shape[0]
    g_inv = simplify(g.inv())
    gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                s = 0
                for sig in range(n):
                    term = (
                        sp.diff(g[nu, sig], coords[mu])
                        + sp.diff(g[mu, sig], coords[nu])
                        - sp.diff(g[mu, nu], coords[sig])
                    )
                    s += g_inv[lam, sig] * term
                gamma[lam][mu][nu] = simplify(Rational(1, 2) * s)
    return gamma, g_inv


def ricci_tensor(gamma: list, coords: tuple) -> Matrix:
    """R_{μν} from Christoffel symbols."""
    n = len(coords)
    R = sp.zeros(n)
    for mu in range(n):
        for nu in range(n):
            acc = 0
            for lam in range(n):
                acc += sp.diff(gamma[lam][mu][nu], coords[lam])
                acc -= sp.diff(gamma[lam][mu][lam], coords[nu])
                for sig in range(n):
                    acc += gamma[lam][sig][lam] * gamma[sig][mu][nu]
                    acc -= gamma[lam][sig][nu] * gamma[sig][mu][lam]
            R[mu, nu] = simplify(acc)
    return R


def einstein_tensor(R_mu_nu: Matrix, g: Matrix, g_inv: Matrix) -> tuple[Matrix, sp.Expr]:
    Ric_scalar = simplify_component((g_inv * R_mu_nu).trace())
    G = R_mu_nu - Rational(1, 2) * Ric_scalar * g
    G = G.applyfunc(simplify_component)
    return G, Ric_scalar


def stress_energy(G: Matrix) -> Matrix:
    """T_{μν} = G_{μν}/(8π)."""
    return G.applyfunc(lambda x: simplify_component(x / (8 * sp.pi)))


def nonzero_components(tensor: Matrix) -> list[tuple[tuple[int, int], sp.Expr]]:
    """Return upper-triangle non-zero entries (μ ≤ ν)."""
    out: list[tuple[tuple[int, int], sp.Expr]] = []
    for mu in range(tensor.shape[0]):
        for nu in range(mu, tensor.shape[1]):
            val = simplify_component(tensor[mu, nu])
            if val != 0:
                out.append(((mu, nu), val))
    return out


def export_component(expr: sp.Expr) -> sp.Expr:
    """Expand numerator for additive LaTeX splitting."""
    num, den = sp.fraction(sp.together(expr))
    num = sp.expand(num)
    if den == 1:
        return num
    return num / den


def compact_latex(expr: sp.Expr, *, expanded: bool = False) -> str:
    """Render expression with compact derivative notation for LaTeX."""
    base = export_component(expr) if expanded else simplify_component(expr)
    tex = sp.latex(base)
    replacements = [
        (r"\frac{d^{2}}{d r^{2}} \Phi{\left(r \right)}", r"\Phi''"),
        (r"\frac{d^{2}}{d r^{2}} b{\left(r \right)}", r"b''"),
        (r"\frac{d^{2}}{d r^{2}} \omega{\left(r \right)}", r"\omega''"),
        (r"\frac{d}{d r} \Phi{\left(r \right)}", r"\Phi'"),
        (r"\frac{d}{d r} b{\left(r \right)}", r"b'"),
        (r"\frac{d}{d r} \omega{\left(r \right)}", r"\omega'"),
        (r"\omega^{2}{\left(r \right)}", r"\omega^{2}"),
        (r"\Phi{\left(r \right)}", r"\Phi"),
        (r"b{\left(r \right)}", r"b"),
        (r"\omega{\left(r \right)}", r"\omega"),
        (r"e^{2 \Phi}", r"e^{2\Phi}"),
        (r"e^{\Phi{\left(r \right)}}", r"e^{\Phi}"),
        (r"e^{- 2 \Phi}", r"e^{-2\Phi}"),
    ]
    for old, new in replacements:
        tex = tex.replace(old, new)
    return tex


def _split_tex_terms(tex: str) -> list[str]:
    """Split a LaTeX sum at top-level +/-."""
    parts: list[str] = []
    buf = ""
    depth = 0
    for ch in tex:
        if ch in "({[":
            depth += 1
        elif ch in ")})]":
            depth -= 1
        if depth == 0 and ch in "+-" and buf.strip():
            parts.append(buf.strip())
            buf = ch
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts or [tex]


def _flatten_tex_parts(parts: list[str], max_len: int = 72) -> list[str]:
    """Recursively split any term still too long for a column."""
    out: list[str] = []
    for part in parts:
        if len(part) <= max_len:
            out.append(part)
            continue
        sub = _split_tex_terms(part)
        if len(sub) == 1:
            out.append(part)
        else:
            out.extend(_flatten_tex_parts(sub, max_len))
    return out


def _aligned_term_lines(lhs: str, tex: str) -> str:
    parts = _flatten_tex_parts(_split_tex_terms(tex))
    lines = [f"{lhs} &= {parts[0]}"]
    for part in parts[1:]:
        if not part:
            continue
        sign = part[0]
        rest = part[1:].lstrip()
        lines.append(f"&{sign} {rest}")
    return " \\\\\n".join(lines)


def latex_multiline_aligned(lhs: str, expr: sp.Expr) -> str:
    """Break a long expression into a column-safe aligned block."""
    expanded_tex = compact_latex(expr, expanded=True)
    if len(expanded_tex) > 90:
        parts = _flatten_tex_parts(_split_tex_terms(expanded_tex))
        if len(parts) > 1:
            body = _aligned_term_lines(lhs, expanded_tex)
            return (
                f"\\begin{{equation}}\n\\begin{{aligned}}\n{body}\n"
                f"\\end{{aligned}}\n\\end{{equation}}\n"
            )

    tex = compact_latex(expr)
    num, den = expr.as_numer_denom()

    if den != 1 and len(tex) > 90:
        num_tex = compact_latex(export_component(num), expanded=True)
        den_tex = compact_latex(den)
        parts = _flatten_tex_parts(_split_tex_terms(num_tex))
        first = parts[0]
        lines = [f"{lhs} &= \\frac{{1}}{{{den_tex}}}\\Bigl({first}"]
        for part in parts[1:]:
            if not part:
                continue
            sign = part[0]
            rest = part[1:].lstrip()
            lines.append(f"&{sign} {rest}")
        lines[-1] = lines[-1] + r"\Bigr)"
        body = " \\\\\n".join(lines)
        return (
            f"\\begin{{equation}}\n\\begin{{aligned}}\n{body}\n"
            f"\\end{{aligned}}\n\\end{{equation}}\n"
        )

    if len(tex) < 90:
        return (
            f"\\begin{{equation}}\n\\begin{{aligned}}\n"
            f"{lhs} &= {tex}\n\\end{{aligned}}\n\\end{{equation}}\n"
        )
    body = _aligned_term_lines(lhs, expanded_tex)
    return f"\\begin{{equation}}\n\\begin{{aligned}}\n{body}\n\\end{{aligned}}\n\\end{{equation}}\n"


def latex_component_align(
    components: list[tuple[tuple[int, int], sp.Expr]],
    label_map: dict[tuple[int, int], str],
    scale: sp.Expr | None = None,
) -> str:
    """Format multiple components, one aligned equation each."""
    blocks: list[str] = []
    for (mu, nu), val in components:
        lbl = label_map.get((mu, nu), f"G_{{{mu}{nu}}}")
        if scale is not None:
            val = simplify_component(val * scale)
        blocks.append(latex_multiline_aligned(lbl, val))
    return "".join(blocks)


def compute_curvature_pipeline(
    equatorial: bool = True, *, explicit_profiles: bool = False
) -> dict:
    """Full Christoffel → Ricci → Einstein → T pipeline."""
    g, coords, funcs = build_metric(equatorial=equatorial)
    gamma, g_inv = christoffel_second_kind(g, coords)
    R_mu_nu = ricci_tensor(gamma, coords)
    G, R_scalar = einstein_tensor(R_mu_nu, g, g_inv)
    T = stress_energy(G)

    if explicit_profiles:
        g = g.applyfunc(substitute_explicit_profiles)
        G = G.applyfunc(substitute_explicit_profiles)
        T = T.applyfunc(substitute_explicit_profiles)
        R_scalar = substitute_explicit_profiles(R_scalar)
        funcs = explicit_metric_profiles()

    return {
        "g": g,
        "g_inv": g_inv,
        "det_g": simplify(g.det()),
        "coords": coords,
        "funcs": funcs,
        "G": G,
        "T": T,
        "R_scalar": R_scalar,
        "explicit_profiles": explicit_profiles,
    }


def energy_density_lab(T: Matrix, g: Matrix) -> sp.Expr:
    """ρ = T_{μν} u^μ u^ν for u^μ = (1,0,0,0) in coordinate frame."""
    u = Matrix([1, 0, 0, 0])
    return simplify_component((u.T * T * u)[0])


def null_quadratic(T: Matrix, k: Matrix) -> sp.Expr:
    return simplify_component((k.T * T * k)[0])


def timelike_quadratic(T: Matrix, u: Matrix) -> sp.Expr:
    return simplify_component((u.T * T * u)[0])
