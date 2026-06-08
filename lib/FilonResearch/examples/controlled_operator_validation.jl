#=
Controlled Operator Framework Validation
=========================================

This example validates that the `ControlledFunctionOp` framework produces
identical results to hand-coded derivative functions.

The Problem
-----------
We solve the same scalar ODE as in `scalar_timedep_convergence.jl`:

    du/dt = λ(t) u,   where λ(t) = α + β·ω·cos(ωt)

ControlledFunctionOp Framework
------------------------------
Instead of manually computing derivatives of A(t), we can express A(t) as a
linear combination of fixed operators with time-dependent scalar coefficients:

    A(t) = Σᵢ cᵢ(t) · Oᵢ

For our scalar case:
    A(t) = α · I + β·ω·cos(ωt) · I
         = c₁(t) · O₁ + c₂(t) · O₂

where O₁ = O₂ = [1] (the 1×1 identity) and:
    c₁(t) = α           (constant)
    c₂(t) = β·ω·cos(ωt) (time-dependent)

The framework then automatically handles the derivative chain rule:
    d^n/dt^n [c(t)·O] = (d^n c/dt^n) · O

This is useful when operators are complex but coefficients are simple functions.

Expected Result
---------------
Both methods should produce numerically identical solutions (within machine precision).
=#

using FilonResearch
using LinearAlgebra

# ---------------------------------------------------------------------------
# Problem Parameters (same as scalar_timedep_convergence.jl)
# ---------------------------------------------------------------------------

s = 1                     # Hermite interpolation order
y0 = ComplexF64[1.0]      # Initial condition
α = 10im                  # Constant part of λ(t)
β = 0.1im                 # Amplitude of cos modulation
ω = 3                     # Modulation frequency
T = 2.0                   # Final time

# Frequency for Filon ansatz
frequencies = [imag(α)]

# ---------------------------------------------------------------------------
# Method 1: Hand-Coded Derivatives
# ---------------------------------------------------------------------------
# Manually computed derivatives of A(t) = [λ(t)]

A_derivs_manual = (
    t -> ComplexF64[α + β * ω * cos(ω * t);;],
    t -> ComplexF64[-β * ω^2 * sin(ω * t);;],
    t -> ComplexF64[-β * ω^3 * cos(ω * t);;],
    t -> ComplexF64[β * ω^4 * sin(ω * t);;],
    t -> ComplexF64[β * ω^5 * cos(ω * t);;],
)

# ---------------------------------------------------------------------------
# Method 2: ControlledFunctionOp Framework
# ---------------------------------------------------------------------------
# Express A(t) = c₁(t)·O₁ + c₂(t)·O₂ with constant operators and
# time-dependent scalar coefficients.

# Operators: both are 1×1 identity matrices
operators = [ComplexF64[1;;], ComplexF64[1;;]]

# Each tuple contains (derivative of c₁, derivative of c₂)
# The framework computes A^(n)(t) = Σᵢ cᵢ^(n)(t) · Oᵢ
A_derivs_controlled = (
    # 0th derivative: c₁ = α, c₂ = β·ω·cos(ωt)
    ControlledFunctionOp(operators, (t -> α, t -> β * ω * cos(ω * t))),
    # 1st derivative: c₁' = 0, c₂' = -β·ω²·sin(ωt)
    ControlledFunctionOp(operators, (t -> 0, t -> -β * ω^2 * sin(ω * t))),
    # 2nd derivative: c₁'' = 0, c₂'' = -β·ω³·cos(ωt)
    ControlledFunctionOp(operators, (t -> 0, t -> -β * ω^3 * cos(ω * t))),
    # 3rd derivative: c₁''' = 0, c₂''' = β·ω⁴·sin(ωt)
    ControlledFunctionOp(operators, (t -> 0, t -> β * ω^4 * sin(ω * t))),
    # 4th derivative: c₁'''' = 0, c₂'''' = β·ω⁵·cos(ωt)
    ControlledFunctionOp(operators, (t -> 0, t -> β * ω^5 * cos(ω * t))),
)

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

nsteps = 1000

num_sol_manual = filon_solve(A_derivs_manual, y0, frequencies, T, nsteps, s)
num_sol_controlled = filon_solve(A_derivs_controlled, y0, frequencies, T, nsteps, s)

max_diff = maximum(abs, num_sol_manual .- num_sol_controlled)

println("Comparing hand-coded derivatives vs. ControlledFunctionOp framework:")
println("  Number of time steps: $nsteps")
println("  Maximum difference:   $max_diff")

if max_diff < 1e-14
    println("  Status: PASS (methods are numerically equivalent)")
else
    println("  Status: FAIL (unexpected discrepancy)")
end
