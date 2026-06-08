#=
Multi-Frequency System Example
==============================

This example demonstrates Filon's method on a non-scalar system with multiple
distinct frequencies. We solve a 3D linear ODE:

    du/dt = A(t) u

where A(t) has three different imaginary eigenvalues (frequencies) plus an
optional time-dependent control perturbation.

Physical Motivation
-------------------
This setup models systems like:
  - Quantum systems with multiple energy levels (each with its own transition frequency)
  - Coupled oscillators with different natural frequencies
  - Multi-mode optical systems

The Challenge
-------------
Standard ODE solvers struggle with highly oscillatory systems because they
must resolve all oscillations with small time steps. Filon's method factors
out the known oscillatory behavior, allowing larger time steps.

Problem Structure
-----------------
    A(t) = D + ε·cos(t)·C

where:
  - D = diag(iω₁, iω₂, iω₃) is the diagonal "drift" Hamiltonian
  - C is a dense "control" matrix representing coupling between modes
  - ε controls the strength of the time-dependent perturbation

When ε = 0, the system decouples and each component oscillates independently.
When ε > 0, the time-dependent control couples the modes.
=#

using FilonResearch
using LinearAlgebra
using OrdinaryDiffEq

# ---------------------------------------------------------------------------
# Problem Parameters
# ---------------------------------------------------------------------------

# Three distinct frequencies (imaginary eigenvalues of the drift Hamiltonian)
ω1 = 10
ω2 = 15
ω3 = 20

# Strength of the time-dependent control (set to 0 for decoupled case)
# Note: Larger values make the problem more challenging
control_strength = 0.00

# Control matrix (dense coupling between all modes)
control_mat = [
    1 4 7
    2 5 8
    3 6 9
]

# Full coefficient matrix: A(t) = D + ε·cos(t)·C
function A(t)
    D = Diagonal([ω1 * im, ω2 * im, ω3 * im])
    return D + cos(t) * control_strength * control_mat
end

# Initial condition
u0 = ComplexF64[1, 2, 3]

# Time span
tspan = (0.0, 10.0)

# ---------------------------------------------------------------------------
# Reference Solution via OrdinaryDiffEq
# ---------------------------------------------------------------------------
# Use a high-accuracy standard ODE solver to generate a reference solution.

function du!(du, u, p, t)
    du .= A(t) * u
end

prob = ODEProblem(du!, u0, tspan)
ref_sol = solve(prob, Tsit5(); abstol=1e-10, reltol=1e-10)

println("Reference solution computed via OrdinaryDiffEq (Tsit5)")
println("  Final time: $(tspan[2])")
println("  Reference u(T): $(ref_sol(tspan[2]))")

# ---------------------------------------------------------------------------
# Filon Solution Setup
# ---------------------------------------------------------------------------
# Express A(t) using the ControlledFunctionOp framework:
#   A(t) = 1·D + ε·cos(t)·C
#
# The derivatives cycle through cos/sin:
#   A⁽⁰⁾(t) = D + ε·cos(t)·C
#   A⁽¹⁾(t) = -ε·sin(t)·C
#   A⁽²⁾(t) = -ε·cos(t)·C
#   A⁽³⁾(t) = ε·sin(t)·C
#   A⁽⁴⁾(t) = ε·cos(t)·C
#   ...

# Operators: D (drift) and C (control)
D_operator = Array(Diagonal([ω1 * im, ω2 * im, ω3 * im]))
C_operator = control_mat

operators = [D_operator, C_operator]

# Derivatives of the scalar coefficients
# Format: (coeff for D, coeff for C)
A_derivs = (
    ControlledFunctionOp(operators, (t -> 1, t -> control_strength * cos(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> -control_strength * sin(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> -control_strength * cos(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> control_strength * sin(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> control_strength * cos(t))),
)

# Frequencies for the Filon ansatz (one per component)
frequencies = [ω1, ω2, ω3]

# Create the Filon problem structure
filon_prob = FilonProblem(u0, tspan[2], frequencies, A_derivs)

# ---------------------------------------------------------------------------
# Solve with Filon's Method
# ---------------------------------------------------------------------------

nsteps = 10          # Number of time steps
hermite_order = 1    # Hermite interpolation order (s)

filon_sol = filon_solve(filon_prob, nsteps, hermite_order)

# Extract final value (last column of solution matrix)
filon_final = filon_sol[:, end]

println("\nFilon solution:")
println("  Number of steps: $nsteps")
println("  Hermite order s: $hermite_order")
println("  Filon u(T): $filon_final")

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

ref_final = ref_sol(tspan[2])
error = norm(filon_final - ref_final)

println("\nComparison:")
println("  ||Filon - Reference||₂ = $error")

# Note: With control_strength = 0, the system decouples and Filon should be
# extremely accurate even with few steps. Increasing control_strength or
# reducing nsteps will increase the error.
