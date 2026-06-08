#=
Scalar Time-Dependent Convergence Test
=======================================

This example demonstrates the Filon method's convergence on a scalar ODE with
a time-dependent coefficient. We solve:

    du/dt = λ(t) u,   u(0) = 1

where λ(t) = α + β·ω·cos(ωt), which has the exact solution:

    u(t) = exp(αt + β·sin(ωt))

The time-dependence in λ(t) requires providing derivatives of the coefficient
matrix A(t) = [λ(t)] up to order 2s (where s is the Hermite interpolation order).

Key concepts demonstrated:
  - Setting up a time-dependent scalar problem
  - Providing hand-computed derivatives of A(t)
  - Measuring convergence order via successive refinement
  - Plotting numerical vs. true solutions
=#

using FilonResearch
using Plots
using LinearAlgebra

# ---------------------------------------------------------------------------
# Problem Parameters
# ---------------------------------------------------------------------------

# Hermite interpolation order (controls smoothness of approximation)
s = 5
rescale = true

# Initial condition (wrapped in array for matrix formulation)
y0 = ComplexF64[1.0]

# Coefficients defining λ(t) = α + β·ω·cos(ωt)
α = 10im      # Constant (imaginary → oscillatory behavior)
β = 0.1im     # Amplitude of time-dependent modulation
ω = 3         # Angular frequency of modulation

# Exact solution: u(t) = u₀·exp(αt + β·sin(ωt))
y(t) = y0 * exp(α * t + β * sin(ω * t))

# Final time for integration
T = 2.0

# ---------------------------------------------------------------------------
# Derivatives of the Coefficient Matrix A(t)
# ---------------------------------------------------------------------------
# Filon's method requires derivatives of A(t) = [λ(t)] up to order 2s.
# For λ(t) = α + β·ω·cos(ωt), the derivatives cycle through sin/cos:
#   λ⁽⁰⁾(t) = α + βω·cos(ωt)
#   λ⁽¹⁾(t) = -βω²·sin(ωt)
#   λ⁽²⁾(t) = -βω³·cos(ωt)
#   λ⁽³⁾(t) = βω⁴·sin(ωt)
#   λ⁽⁴⁾(t) = βω⁵·cos(ωt)
#
# Each derivative is wrapped in a 1×1 matrix (using [x;;] syntax).

A_derivs = (
    t -> ComplexF64[α + β * ω * cos(ω * t);;],
    t -> ComplexF64[-β * ω^2 * sin(ω * t);;],
    t -> ComplexF64[-β * ω^3 * cos(ω * t);;],
    t -> ComplexF64[β * ω^4 * sin(ω * t);;],
    t -> ComplexF64[β * ω^5 * cos(ω * t);;],
    t -> ComplexF64[-β * ω^6 * sin(ω * t);;],
    t -> ComplexF64[-β * ω^7 * cos(ω * t);;],
)

# Frequency for Filon ansatz: u(t) ≈ f(t)·exp(iωt)
# We use imag(α) since α = 10im is the dominant oscillatory term
#frequencies = [imag(α)]
frequencies = [0]

# ---------------------------------------------------------------------------
# Convergence Study
# ---------------------------------------------------------------------------
# Refine the time step by powers of 2 and measure how the error decreases.
# For order-p convergence, halving the step size should reduce error by 2^p.

errors = Float64[]
nsteps_vec = [2^i for i in 0:10]

@show s α β ω rescale frequencies
for nsteps in nsteps_vec
    ts = LinRange(0, T, 1 + nsteps)
    num_sol = filon_solve(A_derivs, y0, frequencies, T, nsteps, s, rescale=rescale)
    true_sol = reduce(hcat, y.(ts))
    error = sum(abs, num_sol - true_sol) / (1+nsteps)
    push!(errors, error)
end

# Compute convergence orders: log₂(error_n / error_{n+1})
error_ratios = [errors[i] / errors[i+1] for i in 1:length(errors)-1]
cvg_orders = log2.(error_ratios)

println("Errors (2 sig. figs):  ", round.(errors, sigdigits=2))
println("Convergence orders:    ", round.(cvg_orders, sigdigits=2))

# ---------------------------------------------------------------------------
# Plot: Numerical vs. True Solution
# ---------------------------------------------------------------------------

pl_nsteps = 1024
pl_ts = LinRange(0, T, 1 + pl_nsteps)
true_sol = [y(t)[1] for t in pl_ts]

pl = plot(pl_ts, real.(true_sol);
    label="True Solution",
    xlabel="t",
    ylabel="Re(u)",
    title="Scalar Time-Dependent ODE: Filon vs. Exact",
    linewidth=2
)

num_sol = filon_solve(A_derivs, y0, frequencies, T, pl_nsteps, s, rescale=rescale)[1, :]
plot!(pl, pl_ts, real.(num_sol);
    label="Filon Solution",
    linestyle=:dash,
    linewidth=2
)

display(pl)
