#=
Filon Method Convergence Order Test
====================================

This script verifies the order of convergence of the Filon quadrature method
for solving oscillatory ODEs. We test:

1. Scalar Dahlquist equation: du/dt = λu with constant λ
2. Matrix system: du/dt = Au with constant A
3. Time-dependent scalar: du/dt = λ(t)u

For Hermite interpolation order s, the expected convergence order is 2s+2.
=#

using FilonResearch
using LinearAlgebra
using Printf

# ============================================================================
# Helper Functions
# ============================================================================

"""
Compute the observed convergence order from a sequence of errors.
Returns the geometric mean of successive error ratios in log scale.
"""
function compute_convergence_order(errors, refinement_factor=2)
    orders = Float64[]
    for i in 1:length(errors)-1
        if errors[i] > 0 && errors[i+1] > 0
            order = log(errors[i] / errors[i+1]) / log(refinement_factor)
            push!(orders, order)
        end
    end
    return orders
end

"""
Print a formatted convergence table.
"""
function print_convergence_table(nsteps_vec, errors, orders; title="")
    println("\n", "="^70)
    println(title)
    println("="^70)
    println(@sprintf("%10s  %15s  %15s", "nsteps", "error", "order"))
    println("-"^45)
    for i in eachindex(nsteps_vec)
        if i == 1
            println(@sprintf("%10d  %15.3e  %15s", nsteps_vec[i], errors[i], "-"))
        else
            println(@sprintf("%10d  %15.3e  %15.2f", nsteps_vec[i], errors[i], orders[i-1]))
        end
    end
    if length(orders) >= 2
        avg_order = sum(orders[end-min(2,length(orders)-1):end]) / min(3, length(orders))
        println("-"^45)
        println(@sprintf("Average order (last 3): %.2f", avg_order))
    end
end

# ============================================================================
# Test 1: Scalar Dahlquist Equation (Constant λ)
# ============================================================================

"""
Test convergence on the scalar Dahlquist equation:
    du/dt = λu,  u(0) = u₀

Exact solution: u(t) = u₀ exp(λt)

Uses the scalar filon_timestep from scalar_filon.jl
"""
function test_scalar_dahlquist(; λ=1.0+5.0im, ω=5.0, u₀=1.0+0.0im, tf=1.0,
                                  s_values=[0, 1, 2, 3], n_refinements=6)
    println("\n" * "="^70)
    println("TEST 1: Scalar Dahlquist Equation (constant λ)")
    println("="^70)
    println("Parameters: λ = $λ, ω = $ω, u₀ = $u₀, tf = $tf")

    # Exact solution
    u_exact = u₀ * exp(λ * tf)

    nsteps_vec = [2^k for k in 0:n_refinements]

    for s in s_values
        errors = Float64[]

        for nsteps in nsteps_vec
            dt = tf / nsteps
            u = u₀
            for n in 0:nsteps-1
                t_n = n * dt
                t_np1 = (n + 1) * dt
                u = filon_timestep(λ, ω, u, s, t_n, t_np1, rescale=true)
            end
            push!(errors, abs(u - u_exact))
        end

        orders = compute_convergence_order(errors)
        expected_order = 2s + 2
        print_convergence_table(nsteps_vec, errors, orders,
            title="s = $s (expected order: $expected_order)")
    end
end

# ============================================================================
# Test 2: Matrix System (Constant A)
# ============================================================================

"""
Test convergence on a matrix ODE system:
    du/dt = Au,  u(0) = u₀

Exact solution: u(t) = exp(At) u₀

Uses filon_solve from filon_timestep.jl
"""
function test_matrix_system(; N=2, n_refinements=6, s_values=[0, 1, 2, 3])
    println("\n" * "="^70)
    println("TEST 2: Matrix System (constant A)")
    println("="^70)

    # Construct a test matrix with known eigenstructure
    # A = i*diag(ω₁, ω₂, ...) + small perturbation
    frequencies = Float64[1.0, 2.0]
    A = Diagonal(im .* frequencies) + 0.1im * ones(N, N)
    A = ComplexF64.(A)

    u₀ = ones(ComplexF64, N)
    tf = 1.0

    # Exact solution via matrix exponential
    u_exact = exp(A * tf) * u₀

    println("Matrix A:")
    display(A)
    println("\nFrequencies for ansatz: $frequencies")
    println("Initial condition: u₀ = $u₀")
    println("Final time: tf = $tf")

    nsteps_vec = [2^k for k in 0:n_refinements]

    for s in s_values
        errors = Float64[]

        for nsteps in nsteps_vec
            sol = filon_solve(A, u₀, frequencies, tf, nsteps, s, rescale=true)
            u_final = sol[:, end]
            push!(errors, norm(u_final - u_exact))
        end

        orders = compute_convergence_order(errors)
        expected_order = 2s + 2
        print_convergence_table(nsteps_vec, errors, orders,
            title="s = $s (expected order: $expected_order)")
    end
end

# ============================================================================
# Test 3: Diagonal System (Should Be Exact)
# ============================================================================

"""
Test that the Filon method is exact for a diagonal system where the
frequencies match the eigenvalues exactly.

    du/dt = i*Ω*u  where Ω = diag(ω₁, ω₂, ...)

With frequencies = [ω₁, ω₂, ...], the ansatz u(t) = f(t)exp(iωt) should
yield f(t) = constant, and the method should be exact.
"""
function test_diagonal_exact()
    println("\n" * "="^70)
    println("TEST 3: Diagonal System (should be exact)")
    println("="^70)

    N = 2
    frequencies = [1.0, 2.0]
    A = Diagonal(im .* frequencies)
    A = ComplexF64.(A)

    u₀ = ComplexF64[1.0, 1.0]
    tf = 5.0  # Large time to stress test

    # Exact solution
    u_exact = [u₀[i] * cis(frequencies[i] * tf) for i in 1:N]

    println("A = i*diag($frequencies)")
    println("u₀ = $u₀, tf = $tf")

    for nsteps in [1, 2, 4, 8]
        for s in [0, 1, 2]
            sol = filon_solve(A, u₀, frequencies, tf, nsteps, s, rescale=true)
            u_final = sol[:, end]
            error = norm(u_final - u_exact)
            println(@sprintf("nsteps=%d, s=%d: error = %.2e", nsteps, s, error))
        end
    end
end

# ============================================================================
# Test 4: Time-Dependent Coefficient
# ============================================================================

"""
Test convergence on a scalar ODE with time-dependent coefficient:
    du/dt = λ(t)u,  u(0) = 1

where λ(t) = α + β*ω*cos(ωt), which has exact solution:
    u(t) = exp(αt + β*sin(ωt))
"""
function test_time_dependent(; α=1.0im, β=0.1im, ω_coef=3.0, tf=1.0,
                               s_values=[1, 2, 3], n_refinements=8)
    println("\n" * "="^70)
    println("TEST 4: Time-Dependent Coefficient")
    println("="^70)
    println("λ(t) = α + β*ω*cos(ωt)")
    println("Parameters: α = $α, β = $β, ω = $ω_coef, tf = $tf")

    # Exact solution
    u_exact(t) = exp(α * t + β * sin(ω_coef * t))

    # Initial condition
    y0 = ComplexF64[1.0]

    # Derivatives of A(t) = [λ(t)]
    # λ(t) = α + β*ω*cos(ωt)
    # λ'(t) = -β*ω²*sin(ωt)
    # λ''(t) = -β*ω³*cos(ωt)
    # etc.
    A_derivs = (
        t -> ComplexF64[α + β * ω_coef * cos(ω_coef * t);;],
        t -> ComplexF64[-β * ω_coef^2 * sin(ω_coef * t);;],
        t -> ComplexF64[-β * ω_coef^3 * cos(ω_coef * t);;],
        t -> ComplexF64[β * ω_coef^4 * sin(ω_coef * t);;],
        t -> ComplexF64[β * ω_coef^5 * cos(ω_coef * t);;],
        t -> ComplexF64[-β * ω_coef^6 * sin(ω_coef * t);;],
        t -> ComplexF64[-β * ω_coef^7 * cos(ω_coef * t);;],
        t -> ComplexF64[β * ω_coef^8 * sin(ω_coef * t);;],
    )

    frequencies = [0.0]  # No oscillatory ansatz

    nsteps_vec = [2^k for k in 0:n_refinements]

    for s in s_values
        if s >= length(A_derivs)
            println("Skipping s=$s: not enough derivatives provided")
            continue
        end

        errors = Float64[]

        for nsteps in nsteps_vec
            sol = filon_solve(A_derivs, y0, frequencies, tf, nsteps, s, rescale=true)
            u_final = sol[1, end]
            u_true = u_exact(tf)
            push!(errors, abs(u_final - u_true))
        end

        orders = compute_convergence_order(errors)
        expected_order = 2s + 2
        print_convergence_table(nsteps_vec, errors, orders,
            title="s = $s (expected order: $expected_order)")
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("Filon Method Convergence Order Verification")
    println("Expected convergence order: 2s + 2 (for Hermite order s)")

    test_scalar_dahlquist()
    test_matrix_system()
    test_diagonal_exact()
    test_time_dependent()

    println("\n" * "="^70)
    println("All tests completed.")
    println("="^70)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
