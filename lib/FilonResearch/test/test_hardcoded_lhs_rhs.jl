using FilonResearch, Test, LinearAlgebra, Printf

# ============================================================================
# Test problems
# ============================================================================

# Problem 1: Constant-coefficient manufactured solution (equal frequencies)
# du/dt = [iω 0; 1 iω] u,  u(0) = [1, 1]
# Solution: u = [1; 1+t] * exp(iωt)
# Envelope is degree-1 polynomial => exact for s >= 0

function make_constant_coeff_problem(ω)
    A = ComplexF64[im*ω 0; 1 im*ω]
    dA = zeros(ComplexF64, 2, 2)
    ddA = zeros(ComplexF64, 2, 2)
    frequencies = [ω, ω]
    u0 = ComplexF64[1.0, 1.0]
    exact(t) = [1.0; 1.0 + t] .* cis(ω * t)
    return (; A, dA, ddA, frequencies, u0, exact,
              A_func = t -> A, dA_func = t -> dA, ddA_func = t -> ddA)
end

# Problem 2: Variable-coefficient system (non-polynomial A, not solvable exactly)
# du/dt = [iω cos(t); -cos(t) iω] u
# Used for convergence-rate testing

function make_variable_coeff_problem(ω)
    A_func(t) = ComplexF64[im*ω cos(t); -cos(t) im*ω]
    dA_func(t) = ComplexF64[0 -sin(t); sin(t) 0]
    ddA_func(t) = ComplexF64[0 -cos(t); cos(t) 0]
    frequencies = [ω, ω]
    u0 = ComplexF64[1.0, 0.0]
    return (; A_func, dA_func, ddA_func, frequencies, u0)
end

# Problem 3: Mixed-frequency system (non-uniform Ω)
# du/dt = [iω₁ cos(t); -cos(t) iω₂] u

function make_mixed_freq_problem(ω1, ω2)
    A_func(t) = ComplexF64[im*ω1 cos(t); -cos(t) im*ω2]
    dA_func(t) = ComplexF64[0 -sin(t); sin(t) 0]
    ddA_func(t) = ComplexF64[0 -cos(t); cos(t) 0]
    frequencies = [ω1, ω2]
    u0 = ComplexF64[1.0, 0.0]
    return (; A_func, dA_func, ddA_func, frequencies, u0)
end

# Problem 4: Larger system (4×4, constant coefficient)
# Block-diagonal with subdiagonal coupling

function make_4x4_problem(ω)
    A = ComplexF64[
        im*ω   0       0       0;
        1       im*ω   0       0;
        0       0       im*ω   0;
        0       0       1       im*ω
    ]
    dA = zeros(ComplexF64, 4, 4)
    ddA = zeros(ComplexF64, 4, 4)
    frequencies = fill(ω, 4)
    u0 = ComplexF64[1.0, 1.0, 1.0, 1.0]
    # Solution for this block structure:
    exact(t) = [1.0; 1.0+t; 1.0; 1.0+t] .* cis(ω * t)
    return (; A, dA, ddA, frequencies, u0, exact,
              A_func = t -> A, dA_func = t -> dA, ddA_func = t -> ddA)
end

# ============================================================================
# Helper: compute reference solution using existing filon_solve
# ============================================================================

function reference_solution(A_funcs::Tuple, u0, frequencies, T, s)
    sol = filon_solve(A_funcs, u0, frequencies, T, 10000, s)
    return sol[:, end]
end

# ============================================================================
# Helper: run a single-method solve loop
# ============================================================================

function solve_loop_s0(A_func, u0, frequencies, T, nsteps; method=:backslash)
    dt = T / nsteps
    u_n = convert(Vector{ComplexF64}, u0)
    for n in 1:nsteps
        t_n = dt * (n - 1)
        t_np1 = dt * n
        A_n = A_func(t_n)
        A_np1 = A_func(t_np1)
        if method == :backslash
            u_n = filon_timestep_s0_backslash(A_n, A_np1, u_n, frequencies, t_n, t_np1)
        else
            u_n = filon_timestep_s0_gmres(A_n, A_np1, u_n, frequencies, t_n, t_np1)
        end
    end
    return u_n
end

function solve_loop_s1(A_func, dA_func, u0, frequencies, T, nsteps; method=:backslash)
    dt = T / nsteps
    u_n = convert(Vector{ComplexF64}, u0)
    for n in 1:nsteps
        t_n = dt * (n - 1)
        t_np1 = dt * n
        if method == :backslash
            u_n = filon_timestep_s1_backslash(
                A_func(t_n), dA_func(t_n), A_func(t_np1), dA_func(t_np1),
                u_n, frequencies, t_n, t_np1)
        else
            u_n = filon_timestep_s1_gmres(
                A_func(t_n), dA_func(t_n), A_func(t_np1), dA_func(t_np1),
                u_n, frequencies, t_n, t_np1)
        end
    end
    return u_n
end

function solve_loop_s2(A_func, dA_func, ddA_func, u0, frequencies, T, nsteps; method=:backslash)
    dt = T / nsteps
    u_n = convert(Vector{ComplexF64}, u0)
    for n in 1:nsteps
        t_n = dt * (n - 1)
        t_np1 = dt * n
        if method == :backslash
            u_n = filon_timestep_s2_backslash(
                A_func(t_n), dA_func(t_n), ddA_func(t_n),
                A_func(t_np1), dA_func(t_np1), ddA_func(t_np1),
                u_n, frequencies, t_n, t_np1)
        else
            u_n = filon_timestep_s2_gmres(
                A_func(t_n), dA_func(t_n), ddA_func(t_n),
                A_func(t_np1), dA_func(t_np1), ddA_func(t_np1),
                u_n, frequencies, t_n, t_np1)
        end
    end
    return u_n
end

# ============================================================================
# Helper: check convergence order
# ============================================================================

function compute_orders(errors)
    return [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
end

function check_convergence_order(errors, expected_order; floor=1e-12)
    orders = compute_orders(errors)
    meaningful = [o for (o, e) in zip(orders, errors[2:end]) if e > floor]
    if isempty(meaningful)
        @test maximum(errors) < floor
    else
        n = min(3, length(meaningful))
        avg_order = sum(meaningful[end-n+1:end]) / n
        @test avg_order > expected_order - 0.5
    end
    @test errors[end] < errors[1] || errors[end] < floor
end

# ============================================================================
# Tests
# ============================================================================

ω = 10.0
T = 1.0
nsteps_vec = [10, 20, 40, 80, 160]

# --------------------------------------------------------------------------
# 1. Exactness: constant-coefficient manufactured solution
# --------------------------------------------------------------------------

@testset "Exactness: constant-coeff manufactured solution" begin
    prob = make_constant_coeff_problem(ω)
    u_exact = prob.exact(T)
    tol = 1e-10

    for (label, method) in [(:backslash, :backslash), (:gmres, :gmres)]
        @testset "s=0 $label" begin
            u = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, 10; method)
            @test maximum(abs.(u - u_exact)) < tol
        end
        @testset "s=1 $label" begin
            u = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, 10; method)
            @test maximum(abs.(u - u_exact)) < tol
        end
        @testset "s=2 $label" begin
            u = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, 10; method)
            @test maximum(abs.(u - u_exact)) < tol
        end
    end
end

# --------------------------------------------------------------------------
# 2. Exactness: 4×4 constant-coefficient system
# --------------------------------------------------------------------------

@testset "Exactness: 4x4 constant-coeff system" begin
    prob = make_4x4_problem(ω)
    u_exact = prob.exact(T)
    tol = 1e-10

    for (label, method) in [(:backslash, :backslash), (:gmres, :gmres)]
        @testset "s=0 $label" begin
            u = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, 10; method)
            @test maximum(abs.(u - u_exact)) < tol
        end
        @testset "s=1 $label" begin
            u = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, 10; method)
            @test maximum(abs.(u - u_exact)) < tol
        end
        @testset "s=2 $label" begin
            u = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, 10; method)
            @test maximum(abs.(u - u_exact)) < tol
        end
    end
end

# --------------------------------------------------------------------------
# 3. Convergence order: variable-coefficient (uniform frequencies)
# --------------------------------------------------------------------------

@testset "Convergence order: variable-coeff, uniform freq" begin
    prob = make_variable_coeff_problem(ω)
    u_ref = reference_solution(
        (prob.A_func, prob.dA_func, prob.ddA_func),
        prob.u0, prob.frequencies, T, 2)

    for (label, method) in [(:backslash, :backslash), (:gmres, :gmres)]
        @testset "s=0 $label (order 2)" begin
            errors = Float64[]
            for nsteps in nsteps_vec
                u = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, nsteps; method)
                push!(errors, maximum(abs.(u - u_ref)))
            end
            check_convergence_order(errors, 2.0)
        end

        @testset "s=1 $label (order 4)" begin
            errors = Float64[]
            for nsteps in nsteps_vec
                u = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, nsteps; method)
                push!(errors, maximum(abs.(u - u_ref)))
            end
            check_convergence_order(errors, 4.0)
        end

        @testset "s=2 $label (order 6)" begin
            errors = Float64[]
            for nsteps in nsteps_vec
                u = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, nsteps; method)
                push!(errors, maximum(abs.(u - u_ref)))
            end
            check_convergence_order(errors, 6.0)
        end
    end
end

# --------------------------------------------------------------------------
# 4. Convergence order: mixed frequencies
# --------------------------------------------------------------------------

@testset "Convergence order: mixed freq" begin
    prob = make_mixed_freq_problem(10.0, 7.0)
    u_ref = reference_solution(
        (prob.A_func, prob.dA_func, prob.ddA_func),
        prob.u0, prob.frequencies, T, 2)

    for (label, method) in [(:backslash, :backslash), (:gmres, :gmres)]
        @testset "s=0 $label (order 2)" begin
            errors = Float64[]
            for nsteps in nsteps_vec
                u = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, nsteps; method)
                push!(errors, maximum(abs.(u - u_ref)))
            end
            check_convergence_order(errors, 2.0)
        end

        @testset "s=1 $label (order 4)" begin
            errors = Float64[]
            for nsteps in nsteps_vec
                u = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, nsteps; method)
                push!(errors, maximum(abs.(u - u_ref)))
            end
            check_convergence_order(errors, 4.0)
        end

        @testset "s=2 $label (order 6)" begin
            errors = Float64[]
            for nsteps in nsteps_vec
                u = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, nsteps; method)
                push!(errors, maximum(abs.(u - u_ref)))
            end
            check_convergence_order(errors, 6.0)
        end
    end
end

# --------------------------------------------------------------------------
# 5. Agreement with existing filon_solve (s=0, 1, 2)
# --------------------------------------------------------------------------

@testset "Agreement with existing filon_solve" begin
    prob = make_variable_coeff_problem(ω)
    nsteps = 40
    tol = 1e-11

    @testset "s=0" begin
        u_new = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
        sol_old = filon_solve((prob.A_func,), prob.u0, prob.frequencies, T, nsteps, 0)
        u_old = sol_old[:, end]
        @test maximum(abs.(u_new - u_old)) < tol
    end

    @testset "s=1" begin
        u_new = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
        sol_old = filon_solve((prob.A_func, prob.dA_func), prob.u0, prob.frequencies, T, nsteps, 1)
        u_old = sol_old[:, end]
        @test maximum(abs.(u_new - u_old)) < tol
    end

    @testset "s=2" begin
        u_new = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
        sol_old = filon_solve((prob.A_func, prob.dA_func, prob.ddA_func), prob.u0, prob.frequencies, T, nsteps, 2)
        u_old = sol_old[:, end]
        @test maximum(abs.(u_new - u_old)) < tol
    end
end

# --------------------------------------------------------------------------
# 6. Backslash vs GMRES agreement
# --------------------------------------------------------------------------

@testset "Backslash vs GMRES agreement" begin
    prob = make_variable_coeff_problem(ω)
    nsteps = 40
    tol = 1e-11

    @testset "s=0" begin
        u_bs = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
        u_gm = solve_loop_s0(prob.A_func, prob.u0, prob.frequencies, T, nsteps; method=:gmres)
        @test maximum(abs.(u_bs - u_gm)) < tol
    end

    @testset "s=1" begin
        u_bs = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
        u_gm = solve_loop_s1(prob.A_func, prob.dA_func, prob.u0, prob.frequencies, T, nsteps; method=:gmres)
        @test maximum(abs.(u_bs - u_gm)) < tol
    end

    @testset "s=2" begin
        u_bs = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
        u_gm = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, nsteps; method=:gmres)
        @test maximum(abs.(u_bs - u_gm)) < tol
    end
end

# --------------------------------------------------------------------------
# 7. filon_solve_hardcoded convenience function
# --------------------------------------------------------------------------

@testset "filon_solve_hardcoded" begin
    prob = make_variable_coeff_problem(ω)
    nsteps = 40
    tol = 1e-11

    for s in 0:2
        funcs = s == 0 ? (prob.A_func,) :
                s == 1 ? (prob.A_func, prob.dA_func) :
                         (prob.A_func, prob.dA_func, prob.ddA_func)
        @testset "s=$s" begin
            u_saves_bs = filon_solve_hardcoded(funcs, prob.u0, prob.frequencies, T, nsteps, s; method=:backslash)
            u_saves_gm = filon_solve_hardcoded(funcs, prob.u0, prob.frequencies, T, nsteps, s; method=:gmres)
            @test maximum(abs.(u_saves_bs[end] - u_saves_gm[end])) < tol
            @test length(u_saves_bs) == nsteps + 1
        end
    end
end

# --------------------------------------------------------------------------
# 8. Agreement with existing filon_solve for mixed frequencies, s=2
# --------------------------------------------------------------------------

@testset "Agreement with existing filon_solve: mixed freq, s=2" begin
    prob = make_mixed_freq_problem(10.0, 7.0)
    nsteps = 40
    tol = 1e-11

    u_new = solve_loop_s2(prob.A_func, prob.dA_func, prob.ddA_func, prob.u0, prob.frequencies, T, nsteps; method=:backslash)
    sol_old = filon_solve((prob.A_func, prob.dA_func, prob.ddA_func), prob.u0, prob.frequencies, T, nsteps, 2)
    u_old = sol_old[:, end]
    @test maximum(abs.(u_new - u_old)) < tol
end
