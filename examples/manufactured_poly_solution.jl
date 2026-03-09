#=
Manufactured Polynomial Solution Tests
=======================================

Tests the Filon method on a system whose exact solution is p(t)·e^(iωt),
where p(t) is a polynomial. Three checks:

1. Exactness: single-timestep error is zero when degree ≤ 2s+1 (Filon) or 2(s+1) (Hermite)
2. Convergence with exact Filon frequencies
3. Convergence with zero frequencies (Hermite treating oscillatory solution as generic)
=#

using FilonResearch
using LinearAlgebra
using Printf

# ============================================================================
# Parameters
# ============================================================================

ω = 10.0
T = 1.0

# ============================================================================
# Helper
# ============================================================================

function print_convergence_table(nsteps_vec, errors, orders; title="")
    println("\n", "="^60)
    println(title)
    println("="^60)
    println(@sprintf("%10s  %15s  %10s", "nsteps", "error", "order"))
    println("-"^40)
    for i in eachindex(nsteps_vec)
        if i == 1
            println(@sprintf("%10d  %15.3e  %10s", nsteps_vec[i], errors[i], "-"))
        else
            println(@sprintf("%10d  %15.3e  %10.2f", nsteps_vec[i], errors[i], orders[i-1]))
        end
    end
    if length(orders) >= 2
        avg_order = sum(orders[end-min(2,length(orders)-1):end]) / min(3, length(orders))
        println("-"^40)
        println(@sprintf("Average order (last 3): %.2f", avg_order))
    end
end

# ============================================================================
# Part 1: Exactness Check (Single Timestep)
# ============================================================================

println("="^60)
println("PART 1: Exactness Check (single timestep, T=$T, ω=$ω)")
println("="^60)
println()
println("Filon with parameter s uses Hermite interpolation of degree 2s+1.")
println("  ω≠0 (Filon): exact for polynomial envelope degree ≤ 2s+1")
println("  ω=0 (Hermite): exact for polynomial degree ≤ 2(s+1)")
println()

for s in 0:3
    max_degree = 2*(s+1) + 1  # test up to one past Hermite exactness boundary
    println(@sprintf("s = %d:  Filon exact ≤ %d,  Hermite exact ≤ %d", s, 2s+1, 2*(s+1)))
    println(@sprintf("  %6s  %15s  %15s", "degree", "err(ω=$ω)", "err(ω=0)"))
    println("  ", "-"^40)

    for degree in 0:max_degree
        N = degree + 1

        # Filon (exact frequencies)
        A = poly_osc_ode_mat(ω, degree)
        u0 = ones(ComplexF64, N)
        true_sol = poly_osc_solution(ω, degree, T)
        sol_filon = filon_timestep(A, u0, fill(ω, N), 0.0, T, s)
        err_filon = maximum(abs, sol_filon - true_sol)

        # Hermite (ω=0 system, zero frequencies)
        A0 = poly_osc_ode_mat(0.0, degree)
        true_sol0 = poly_osc_solution(0.0, degree, T)
        sol_hermite = filon_timestep(A0, real.(u0), zeros(N), 0.0, T, s)
        err_hermite = maximum(abs, sol_hermite - true_sol0)

        println(@sprintf("  %6d  %15.3e  %15.3e", degree, err_filon, err_hermite))
    end
    println()
end

# ============================================================================
# Part 2: Convergence with Exact Filon Frequencies
# ============================================================================

println()
println("="^60)
println("PART 2: Convergence (exact Filon frequencies, ω=$ω)")
println("="^60)
println("Solution envelope degree 10, exceeds 2s+1 for all s ≤ 3.")
println("Expected order: 2(s+1)")

degree = 10
N = degree + 1
A = poly_osc_ode_mat(ω, degree)
u0 = ones(ComplexF64, N)
frequencies = fill(ω, N)
true_sol = poly_osc_solution(ω, degree, T)
nsteps_vec = [2^k for k in 1:10]

for s in 0:3
    errors = Float64[]
    for nsteps in nsteps_vec
        sol = filon_solve(A, u0, frequencies, T, nsteps, s)[:, end]
        push!(errors, maximum(abs, sol - true_sol))
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1 if errors[i] > 0 && errors[i+1] > 0]
    print_convergence_table(nsteps_vec, errors, orders,
        title="s = $s (expected order: $(2*(s+1)))")
end

# ============================================================================
# Part 3: Convergence with Zero Frequencies (Hermite on oscillatory solution)
# ============================================================================

println()
println("="^60)
println("PART 3: Convergence (zero frequencies / Hermite, ω=$ω)")
println("="^60)
println("Same oscillatory solution as Part 2, but frequencies = zeros.")
println("Hermite treats p(t)·e^(iωt) as a generic function — never exact.")
println("Expected order: 2(s+1)")

zero_freqs = zeros(N)

for s in 0:3
    errors = Float64[]
    for nsteps in nsteps_vec
        sol = filon_solve(A, u0, zero_freqs, T, nsteps, s)[:, end]
        push!(errors, maximum(abs, sol - true_sol))
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1 if errors[i] > 0 && errors[i+1] > 0]
    print_convergence_table(nsteps_vec, errors, orders,
        title="s = $s (expected order: $(2*(s+1)))")
end

println("\nDone.")
