using FilonResearch, Test, LinearAlgebra
using StaticArrays
import FilonResearch: _filon_step_static, _matderivs, _apply_M!, _opderivs,
    _FilonDynWS, _static_weights, _dynamic_weights

# =============================================================================
# Problem builders — each returns matched static (SMatrix tuple) and dynamic
# (Vector) ControlledOperators for the SAME A(t), plus dense derivative
# closures (used only to build an independent reference solution).
# =============================================================================

# Constant-coefficient manufactured solution (equal frequencies):
#   dψ/dt = [iω 0; 1 iω] ψ,  ψ(0) = [1,1]   ⇒   ψ = [1, 1+t] e^{iωt}.
# Envelope is degree 1, so the Filon method is *exact* for every s ≥ 0.
function constant_coeff_problem(ω)
    A = ComplexF64[im*ω 0; 1 im*ω]
    ctrls = (ConstantControl(1.0),)
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(A),))
    co_dynamic = ControlledOperator(ctrls, [A])
    exact(t) = ComplexF64[1.0, 1.0 + t] .* cis(ω * t)
    return (; co_static, co_dynamic, frequencies = [ω, ω], ψ0 = ComplexF64[1.0, 0.0 + 1.0],
              exact)
end

# Variable-coefficient system (uniform frequencies), not solvable in closed form:
#   dψ/dt = [iω cos t; -cos t iω] ψ = drift + cos(t) * offdiag.
function variable_coeff_problem(ω)
    drift = ComplexF64[im*ω 0; 0 im*ω]
    offd = ComplexF64[0 1; -1 0]
    ctrls = (ConstantControl(1.0), FourierControl(0.0, [1.0], [0.0], 1.0))   # 1, cos(t)
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(drift), SMatrix{2,2}(offd)))
    co_dynamic = ControlledOperator(ctrls, [drift, offd])
    A_func(t) = ComplexF64[im*ω cos(t); -cos(t) im*ω]
    dA_func(t) = ComplexF64[0 -sin(t); sin(t) 0]
    ddA_func(t) = ComplexF64[0 -cos(t); cos(t) 0]
    return (; co_static, co_dynamic, frequencies = [ω, ω], ψ0 = ComplexF64[1.0, 0.0],
              A_func, dA_func, ddA_func)
end

# Mixed-frequency system (non-uniform Ω = diag(ω1, ω2)):
#   dψ/dt = [iω1 cos t; -cos t iω2] ψ.
function mixed_freq_problem(ω1, ω2)
    drift = ComplexF64[im*ω1 0; 0 im*ω2]
    offd = ComplexF64[0 1; -1 0]
    ctrls = (ConstantControl(1.0), FourierControl(0.0, [1.0], [0.0], 1.0))
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(drift), SMatrix{2,2}(offd)))
    co_dynamic = ControlledOperator(ctrls, [drift, offd])
    A_func(t) = ComplexF64[im*ω1 cos(t); -cos(t) im*ω2]
    dA_func(t) = ComplexF64[0 -sin(t); sin(t) 0]
    ddA_func(t) = ComplexF64[0 -cos(t); cos(t) 0]
    return (; co_static, co_dynamic, frequencies = [ω1, ω2], ψ0 = ComplexF64[1.0, 0.0],
              A_func, dA_func, ddA_func)
end

# Independent reference: the existing (separately implemented) filon_solve, run
# fine, returns the final state.
function reference_final(prob, T)
    sol = filon_solve((prob.A_func, prob.dA_func, prob.ddA_func),
                      prob.ψ0, prob.frequencies, T, 10000, 2)
    return sol[:, end]
end

# Empirical convergence order between successive refinements, tolerant of an
# accuracy floor (the reference is itself only good to ~1e-12).
function check_order(errors, expected; floor = 5e-12)
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    meaningful = [o for (o, e) in zip(orders, errors[2:end]) if e > floor]
    if isempty(meaningful)
        @test maximum(errors) < floor                  # already at the floor
    else
        @test maximum(meaningful) > expected - 0.5      # achieves the rate before flooring
    end
    @test errors[end] < errors[1] || errors[end] < floor
end

# Bare globals (not `const`): other test files in the shared runtests session
# already define `T`/`ω` as globals, so a `const` redeclaration would error.
T = 1.0
ω = 10.0
NS = [10, 20, 40, 80]

# =============================================================================
# 1. Exactness on the degree-1 envelope problem (all s, both variants)
# =============================================================================
@testset "Exactness: constant-coeff manufactured solution" begin
    prob = constant_coeff_problem(ω)
    uex = prob.exact(T)
    for s in 0:2, (lbl, co) in (("static", prob.co_static), ("dynamic", prob.co_dynamic))
        u = filon_solve_hardcoded(co, prob.ψ0, prob.frequencies, T / 10, 10, s;
                                  save_final_only = true)
        @test maximum(abs.(u .- uex)) < 1e-10
    end
end

# =============================================================================
# 2. Convergence orders 2/4/6, uniform and mixed frequencies, both variants
# =============================================================================
@testset "Convergence orders" begin
    for (name, prob) in (("variable-coeff, uniform freq", variable_coeff_problem(ω)),
                         ("mixed freq", mixed_freq_problem(10.0, 7.0)))
        uref = reference_final(prob, T)
        @testset "$name" begin
            for s in 0:2, (lbl, co) in (("static", prob.co_static),
                                        ("dynamic", prob.co_dynamic))
                errs = Float64[]
                for ns in NS
                    u = filon_solve_hardcoded(co, prob.ψ0, prob.frequencies, T / ns, ns, s;
                                              save_final_only = true)
                    push!(errs, maximum(abs.(u .- uref)))
                end
                @testset "s=$s ($lbl)" begin
                    check_order(errs, 2s + 2.0)
                end
            end
        end
    end
end

# =============================================================================
# 3. Static and dynamic variants agree (incl. forcing :dynamic on a static co)
# =============================================================================
@testset "Static vs dynamic agreement" begin
    prob = variable_coeff_problem(ω)
    ns = 50
    for s in 0:2
        us = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, T / ns, ns, s;
                                   save_final_only = true)
        ud = filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, T / ns, ns, s;
                                   save_final_only = true)
        # dynamic path applied to the static-layout operator (mul! on tuple matrices)
        uf = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, T / ns, ns, s;
                                   save_final_only = true, variant = :dynamic)
        @test maximum(abs.(us .- ud)) < 1e-11
        @test maximum(abs.(us .- uf)) < 1e-11
    end
end

# =============================================================================
# 3b. GMRES warm-start (dynamic variant) must not change the answer
# =============================================================================
@testset "Dynamic warm-start agreement" begin
    prob = variable_coeff_problem(ω)
    ns = 50
    for s in 0:2
        cold = filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, T / ns, ns, s;
                                     save_final_only = true)
        warm = filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, T / ns, ns, s;
                                     save_final_only = true, warm_start = true)
        @test maximum(abs.(warm .- cold)) < 1e-11
    end
    # warm-start is also accepted (and ignored) by the static variant
    s = 2
    a = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, T / ns, ns, s;
                              save_final_only = true)
    b = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, T / ns, ns, s;
                              save_final_only = true, warm_start = true)
    @test a == b
end

# =============================================================================
# 4. Single-step API matches the driver, and reuses precomputed weights
# =============================================================================
@testset "filon_timestep_hardcoded single step" begin
    prob = variable_coeff_problem(ω)
    Δt = 0.01
    for s in 0:2
        # static
        wps = filon_weight_phases(prob.co_static, prob.frequencies, Δt, s)
        ψ = SVector{2,ComplexF64}(prob.ψ0)
        ψ1 = filon_timestep_hardcoded(prob.co_static, ψ, 0.0, Δt, wps)
        ref = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, Δt, 1, s)
        @test maximum(abs.(Vector(ψ1) .- ref[:, end])) < 1e-12
        # dynamic
        wpd = filon_weight_phases(prob.co_dynamic, prob.frequencies, Δt, s)
        ψ1d = filon_timestep_hardcoded(prob.co_dynamic, Vector(prob.ψ0), 0.0, Δt, wpd)
        @test maximum(abs.(ψ1d .- ref[:, end])) < 1e-11
    end
end

# =============================================================================
# 5. Saving options: history shape, save_every subsampling, final-only
# =============================================================================
@testset "Saving options" begin
    prob = variable_coeff_problem(ω)
    nsteps = 100
    Δt = T / nsteps

    full = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, Δt, nsteps, 2)
    @test size(full) == (2, nsteps + 1)                       # initial + every step

    sub = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, Δt, nsteps, 2;
                                save_every = 10)
    @test size(sub) == (2, length(0:10:nsteps))               # initial + every 10th
    @test sub[:, 1] ≈ full[:, 1]                              # initial state
    @test sub[:, end] ≈ full[:, end]                          # final state preserved
    @test sub[:, 2] ≈ full[:, 11]                             # column 2 ↔ step 10

    # save_every that does not divide nsteps still includes the final step
    sub2 = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, Δt, nsteps, 2;
                                 save_every = 30)
    @test size(sub2, 2) == length(0:30:nsteps) + 1            # 0,30,60,90, +final 100
    @test sub2[:, end] ≈ full[:, end]

    fin = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, Δt, nsteps, 2;
                                save_final_only = true)
    @test fin isa AbstractVector
    @test fin ≈ full[:, end]

    fin_d = filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt, nsteps, 2;
                                  save_final_only = true)
    @test maximum(abs.(fin_d .- fin)) < 1e-11
end

# =============================================================================
# 6. Allocation: the static step and the dynamic matvec are allocation-free,
#    and the static timestepping loop allocates only the one-time precompute.
# =============================================================================
@testset "Allocation" begin
    prob = variable_coeff_problem(ω)
    Δt = 0.01
    for s in 0:2
        vs = Val(s)
        # static step
        wp = _static_weights(prob.co_static, prob.frequencies, Δt, vs, Val(2))
        ψ = SVector{2,ComplexF64}(prob.ψ0)
        An = _matderivs(prob.co_static, 0.0, vs)
        Anp1 = _matderivs(prob.co_static, Δt, vs)
        step() = _filon_step_static(An, Anp1, ψ, wp)
        step()
        @test (@allocated step()) == 0

        # dynamic matvec
        wpd = _dynamic_weights(prob.co_dynamic, prob.frequencies, Δt, vs)
        ws = _FilonDynWS(2)
        x = Vector(prob.ψ0); out = zeros(ComplexF64, 2)
        ops = _opderivs(prob.co_dynamic, 0.0, vs)
        applyM() = _apply_M!(out, x, ops, wpd.WE, wpd.freqs, ws, vs)
        applyM()
        @test (@allocated applyM()) == 0
    end

    # The static solve's total allocation is independent of nsteps (per-step ≈ 0).
    solve(ns) = filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies,
                                      T / ns, ns, 2; save_final_only = true)
    solve(100); solve(800)
    @test (@allocated solve(800)) <= (@allocated solve(100)) + 64
end

# =============================================================================
# 7. Argument validation
# =============================================================================
@testset "Argument validation" begin
    prob = variable_coeff_problem(ω)
    @test_throws ArgumentError filon_solve_hardcoded(prob.co_static, prob.ψ0,
        prob.frequencies, 0.1, 10, 3)                          # s out of range
    @test_throws ArgumentError filon_solve_hardcoded(prob.co_static, prob.ψ0,
        prob.frequencies, 0.1, 0, 1)                           # nsteps < 1
    @test_throws ArgumentError filon_solve_hardcoded(prob.co_static, prob.ψ0,
        prob.frequencies, 0.1, 10, 1; save_every = 0)          # save_every < 1
    @test_throws DimensionMismatch filon_solve_hardcoded(prob.co_static, prob.ψ0,
        [ω], 0.1, 10, 1)                                       # wrong #frequencies
end
