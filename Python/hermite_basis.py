"""
Hermite cardinal basis polynomials and their (oscillatory) integrals.

For an interval [a, b] and a derivative order s, this builds the 2(s+1)
cardinal polynomials

    ell^{[a,b], s}_{E, j}(t)   (E = left node  a, j = 0,...,s)
    ell^{[a,b], s}_{I, j}(t)   (I = right node b, j = 0,...,s)

of degree 2s+1 satisfying

    (d/dt)^k ell^{[a,b], s}_{E, j} (a) = delta_{j,k},  (d/dt)^k ell^{[a,b], s}_{E, j} (b) = 0
    (d/dt)^k ell^{[a,b], s}_{I, j} (b) = delta_{j,k},  (d/dt)^k ell^{[a,b], s}_{I, j} (a) = 0

for k = 0, ..., s.

The endpoints a, b are kept symbolic so the same basis can later be
specialised to [-1, 1], [0, 1], [0, Delta t], etc.
"""

import sympy as sp


def hermite_basis_polynomials(s, a, b, t):
    """Return dict {(node, j): polynomial in t} of cardinal polynomials.

    node is the string "E" (left, t = a) or "I" (right, t = b).
    j ranges over 0..s.
    """
    n = 2 * (s + 1)
    coeffs = sp.symbols(f"c0:{n}")
    poly = sum(c * t**i for i, c in enumerate(coeffs))

    derivs = [sp.diff(poly, t, k) for k in range(s + 1)]

    basis = {}
    for node, alpha, beta in [("E", a, b), ("I", b, a)]:
        for j in range(s + 1):
            eqs = []
            for k in range(s + 1):
                eqs.append(sp.Eq(derivs[k].subs(t, alpha), 1 if k == j else 0))
                eqs.append(sp.Eq(derivs[k].subs(t, beta), 0))
            sol = sp.solve(eqs, coeffs, dict=True)[0]
            basis[(node, j)] = sp.expand(poly.subs(sol))
    return basis


def integrate_basis(basis, t, a, b):
    """Integrate each cardinal polynomial over [a, b]."""
    return {k: sp.simplify(sp.integrate(p, (t, a, b))) for k, p in basis.items()}


def integrate_basis_oscillatory(basis, t, a, b, w):
    """Integrate p(t) * exp(i*w*t) over [a, b] for each cardinal polynomial.

    Result is left in closed form (complex exponential).
    """
    out = {}
    for k, p in basis.items():
        val = sp.integrate(p * sp.exp(sp.I * w * t), (t, a, b))
        out[k] = sp.simplify(val)
    return out


def _label_latex(node, j, s, a, b):
    return (
        f"\\ell^{{[{sp.latex(a)},{sp.latex(b)}],\\,s={s}}}"
        f"_{{{node},\\,j={j}}}(t)"
    )


def _print_align(lines):
    """Print a list of "lhs = rhs" strings inside an align environment."""
    print("\\begin{align}")
    for i, (lhs, rhs) in enumerate(lines):
        sep = " \\\\" if i < len(lines) - 1 else ""
        print(f"    {lhs} &= {rhs}{sep}")
    print("\\end{align}")


def _factor_polynomial(expr, t):
    """Return (N, q) such that expr = q / N with q having integer (or
    polynomial-in-other-symbols) coefficients in t and N a common factor.
    """
    poly = sp.Poly(sp.together(expr), t)
    coeffs = poly.all_coeffs()
    common = sp.S.One
    for c in coeffs:
        _, d = sp.fraction(sp.together(c))
        common = sp.lcm(common, d)
    cleared = sp.expand(sp.together(expr) * common)
    return common, cleared


def _latex_inc(expr):
    """sp.latex with terms ordered by increasing degree."""
    return sp.latex(expr, order="rev-lex")


def _polynomial_latex(expr, t):
    """LaTeX for a polynomial in t, pulling out a single 1/N prefactor."""
    N, q = _factor_polynomial(expr, t)
    if N == 1:
        return _latex_inc(q)
    return f"\\frac{{1}}{{{_latex_inc(N)}}}\\left( {_latex_inc(q)} \\right)"


def _factored_polynomial_with_sign(expr, t):
    """Factor expr over the integers. Returns (sign, latex) where each factor
    in latex starts with a positive constant term; any sign flips needed to
    achieve that are accumulated into sign (+1 or -1).
    """
    content, factor_pairs = sp.factor_list(expr)
    sign = 1
    if content < 0:
        sign = -sign
        content = -content
    factors = []
    for p, m in factor_pairs:
        if p.subs(t, 0) < 0:
            p = sp.expand(-p)
            if m % 2 == 1:
                sign = -sign
        factors.append((p, m))
    parts = []
    if content != 1:
        parts.append(_latex_inc(content))
    for p, m in factors:
        wrapped = f"\\left( {_latex_inc(p)} \\right)"
        parts.append(wrapped if m == 1 else f"{wrapped}^{{{m}}}")
    return sign, " ".join(parts)


def _basis_polynomial_latex(expr, t, node, s, a, b):
    """LaTeX for a Hermite cardinal polynomial.

    Pulls out the vanishing factor at the opposite endpoint -- (b-t)^{s+1}
    for the left node (E) and (t-a)^{s+1} for the right node (I), then
    further factors the degree-s remainder over the integers. Any negative
    sign needed to keep each factor's constant term positive is absorbed
    into the leading 1/N coefficient.
    """
    linear = (b - t) if node == "E" else (t - a)
    vanish = linear ** (s + 1)
    quotient = sp.cancel(expr / vanish)
    N, q = _factor_polynomial(quotient, t)

    vanish_tex = f"\\left( {_latex_inc(linear)} \\right)^{{{s + 1}}}"
    sign = 1
    factored_tex = ""
    if q != 1:
        sign, factored_tex = _factored_polynomial_with_sign(q, t)

    pieces = []
    if N != 1:
        coeff = f"\\frac{{1}}{{{_latex_inc(N)}}}"
        if sign == -1:
            coeff = "-" + coeff
        pieces.append(coeff)
    elif sign == -1:
        pieces.append("-")
    pieces.append(vanish_tex)
    if factored_tex:
        pieces.append(factored_tex)
    return " ".join(pieces)


def _print_basis_section(s, basis, t, a, b):
    print(f"% --- Basis polynomials, s = {s} ---")
    _print_align([
        (
            _label_latex(node, j, s, a, b),
            _basis_polynomial_latex(p, t, node, s, a, b),
        )
        for (node, j), p in basis.items()
    ])


def _print_plain_section(s, basis, t, a, b):
    print(f"% --- Plain integrals, s = {s} ---")
    plain = integrate_basis(basis, t, a, b)
    _print_align([
        (
            f"\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} "
            f"{_label_latex(node, j, s, a, b)}\\,dt",
            _latex_inc(val),
        )
        for (node, j), val in plain.items()
    ])


def _print_oscillatory_section(s, basis, t, a, b, w):
    print(f"% --- Oscillatory integrals, s = {s} ---")
    osc = integrate_basis_oscillatory(basis, t, a, b, w)
    _print_align([
        (
            f"\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} "
            f"{_label_latex(node, j, s, a, b)}\\,e^{{i w t}}\\,dt",
            _latex_inc(val),
        )
        for (node, j), val in osc.items()
    ])


def report_all(s_values, a, b, t, w, *, do_oscillatory=True):
    """Print each output type across all s values, grouped by section.

    For each section (basis polynomials, plain integrals, oscillatory
    integrals), iterate through all s in s_values before moving on to the
    next section.
    """
    print(f"\n% === Hermite cardinal basis, "
          f"interval [{sp.latex(a)}, {sp.latex(b)}] ===\n")
    bases = {s: hermite_basis_polynomials(s, a, b, t) for s in s_values}

    for s in s_values:
        _print_basis_section(s, bases[s], t, a, b)
        print()

    for s in s_values:
        _print_plain_section(s, bases[s], t, a, b)
        print()

    if do_oscillatory:
        for s in s_values:
            _print_oscillatory_section(s, bases[s], t, a, b, w)
            print()

    return bases


if __name__ == "__main__":
    t, w = sp.symbols("t w", real=True)

    # Default demo: interval [-1, 1] with s = 0, 1, 2.
    a, b = sp.Integer(-1), sp.Integer(1)
    report_all((0, 1, 2, 3), a, b, t, w)

    # Example of switching to [0, 1] or [0, Delta t]:
    #
    # dt = sp.symbols("Delta_t", positive=True)
    # report_all((0, 1), sp.Integer(0), dt, t, w)
