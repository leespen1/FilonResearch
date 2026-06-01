module ControlledOperatorsTaylorDiffExt

using ControlledOperators: ControlledOperators, AbstractControl, Derivative, DerivativeUpTo,
                           derivative
using StaticArrays: SVector
using TaylorDiff: TaylorDiff

# Generic Taylor-mode fallback for controls that only implement the `Derivative{0}` evaluation.
# Defined on the *abstract* `AbstractControl`, so a control with its own (more specific) analytic
# derivative — every concrete control in this package — still wins by dispatch.

"""
    derivative(c::AbstractControl, t, ::Derivative{N})

Taylor-mode fallback giving the `N`-th time-derivative of any real-valued control that defines
only the `Derivative{0}` evaluation (TaylorDiff's `TaylorScalar <: Real`).

Note: unlike the built-in controls, this fallback deliberately leaves `t` **untyped** rather
than `t::Real`.  It is generic over *arbitrary* user controls, and a user may write
`derivative(::MyControl, t, ::Derivative{0})` with `t` untyped.  Adding `t::Real` here would
make the two methods mutually ambiguous — the user's is more specific on the control type and
the `Derivative{0}` order, this one would be more specific on `t::Real`, and neither dominates.
Untyped `t` lets the user's method take precedence cleanly.
"""
function ControlledOperators.derivative(c::AbstractControl, t, ::Derivative{N}) where {N}
    g = τ -> derivative(c, τ, Derivative{0}())
    return N == 0 ? g(t) : TaylorDiff.derivative(g, t, Val(N))
end

"""
    derivative(c::AbstractControl, t, ::DerivativeUpTo{N})

Taylor-mode fallback returning all derivatives `0..N` of a `Derivative{0}`-only control in a
single pass.  Like the `Derivative{N}` method above, `t` is left **untyped** (not `t::Real`) to
avoid a dispatch ambiguity with user controls that leave `t` untyped — see that method's
docstring.

Implementation: `TaylorDiff.flatten` returns the Taylor coefficients `(f, f'/1!, …, fᴺ/N!)`;
multiplying entry `m` (order `m-1`) by `(m-1)!` recovers the derivative.  Using `flatten` rather
than a per-order `Val`-indexed `extract_derivative` keeps the result type-stable — the order
index never enters a type parameter.
"""
function ControlledOperators.derivative(c::AbstractControl, t, ::DerivativeUpTo{N}) where {N}
    g = τ -> derivative(c, τ, Derivative{0}())
    ts = TaylorDiff.derivatives(g, t, one(t), Val(N))
    coeffs = TaylorDiff.flatten(ts)
    return SVector(ntuple(m -> coeffs[m] * factorial(m - 1), Val(N + 1)))
end

end # module
