using FilonResearch, Test, LinearAlgebra
using StaticArrays
import FilonResearch: _hermite_step_static, _matderivs, _apply_M_hermite!, _opderivs,
    _FilonDynWS, _static_hermite_weights, _dynamic_hermite_weights

# =============================================================================
# Problem builders — matched static (SMatrix tuple) and dynamic (Vector)
# ControlledOperators for the same A(t).  Mirrors test_hardcoded_filon.jl.
# =============================================================================

# Constant-coefficient manufactured solution:
#   dψ/dt = [0 0; 1 0] ψ, ψ(0)=[1,0] ⇒ ψ = [1, t].  Degree-1 envelope, so the
#   Hermite method is exact for every s ≥ 0.
function hermite_constant_problem()
    A = ComplexF64[0 0; 1 0]
    ctrls = (ConstantControl(1.0),)
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(A),))
    co_dynamic = ControlledOperator(ctrls, [A])
    exact(t) = ComplexF64[1.0, t]
    return (; co_static, co_dynamic, ψ0 = ComplexF64[1.0, 0.0], exact)
end

# Variable-coefficient system (not solvable in closed form):
#   dψ/dt = [0 cos t; -cos t 0] ψ = cos(t) * offdiag.  (Skew ⇒ norm-preserving.)
function hermite_variable_problem()
    drift = ComplexF64[0 0; 0 0]
    offd = ComplexF64[0 1; -1 0]
    ctrls = (ConstantControl(1.0), FourierControl(0.0, [1.0], [0.0], 1.0))   # 1, cos(t)
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(drift), SMatrix{2,2}(offd)))
    co_dynamic = ControlledOperator(ctrls, [drift, offd])
    return (; co_static, co_dynamic, ψ0 = ComplexF64[1.0, 0.0])
end

# Empirical convergence order, tolerant of an accuracy floor.
function herm_check_order(errors, expected; floor = 5e-12)
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    meaningful = [o for (o, e) in zip(orders, errors[2:end]) if e > floor]
    if isempty(meaningful)
        @test maximum(errors) < floor
    else
        @test maximum(meaningful) > expected - 0.5
    end
    @test errors[end] < errors[1] || errors[end] < floor
end

T = 1.0
NS = [10, 20, 40, 80]

# =============================================================================
# 1. Hardcoded Hermite IS the ω = 0 case of hardcoded Filon (the defining check)
# =============================================================================
@testset "Hermite == Filon(ω=0)" begin
    prob = hermite_variable_problem()
    ns = 50
    zerofreqs = [0.0, 0.0]
    for s in 0:2, (lbl, co) in (("static", prob.co_static), ("dynamic", prob.co_dynamic))
        h = hermite_solve_hardcoded(co, prob.ψ0, T / ns, ns, s; save_final_only = true)
        f = filon_solve_hardcoded(co, prob.ψ0, zerofreqs, T / ns, ns, s; save_final_only = true)
        @test maximum(abs.(h .- f)) < 1e-13
    end
end

# =============================================================================
# 2. Exactness on the degree-1 envelope problem (all s, both variants)
# =============================================================================
@testset "Exactness: constant-coeff manufactured solution" begin
    prob = hermite_constant_problem()
    uex = prob.exact(T)
    for s in 0:2, (lbl, co) in (("static", prob.co_static), ("dynamic", prob.co_dynamic))
        u = hermite_solve_hardcoded(co, prob.ψ0, T / 10, 10, s; save_final_only = true)
        @test maximum(abs.(u .- uex)) < 1e-10
    end
end

# =============================================================================
# 3. Convergence orders 2/4/6, both variants
# =============================================================================
@testset "Convergence orders" begin
    prob = hermite_variable_problem()
    uref = hermite_solve_hardcoded(prob.co_static, prob.ψ0, T / 10000, 10000, 2;
                                   save_final_only = true)
    for s in 0:2, (lbl, co) in (("static", prob.co_static), ("dynamic", prob.co_dynamic))
        errs = Float64[]
        for ns in NS
            u = hermite_solve_hardcoded(co, prob.ψ0, T / ns, ns, s; save_final_only = true)
            push!(errs, maximum(abs.(u .- uref)))
        end
        @testset "s=$s ($lbl)" begin
            herm_check_order(errs, 2s + 2.0)
        end
    end
end

# =============================================================================
# 4. Static and dynamic variants agree (incl. forcing :dynamic on a static co)
# =============================================================================
@testset "Static vs dynamic agreement" begin
    prob = hermite_variable_problem()
    ns = 50
    for s in 0:2
        us = hermite_solve_hardcoded(prob.co_static, prob.ψ0, T / ns, ns, s;
                                     save_final_only = true)
        ud = hermite_solve_hardcoded(prob.co_dynamic, prob.ψ0, T / ns, ns, s;
                                     save_final_only = true)
        uf = hermite_solve_hardcoded(prob.co_static, prob.ψ0, T / ns, ns, s;
                                     save_final_only = true, variant = :dynamic)
        @test maximum(abs.(us .- ud)) < 1e-11
        @test maximum(abs.(us .- uf)) < 1e-11
    end
end

# =============================================================================
# 5. Single-step API matches the driver
# =============================================================================
@testset "hermite_timestep_hardcoded single step" begin
    prob = hermite_variable_problem()
    Δt = 0.01
    for s in 0:2
        wps = hermite_weight_phases(prob.co_static, Δt, s)
        ψ = SVector{2,ComplexF64}(prob.ψ0)
        ψ1 = hermite_timestep_hardcoded(prob.co_static, ψ, 0.0, Δt, wps)
        ref = hermite_solve_hardcoded(prob.co_static, prob.ψ0, Δt, 1, s)
        @test maximum(abs.(Vector(ψ1) .- ref[:, end])) < 1e-12

        wpd = hermite_weight_phases(prob.co_dynamic, Δt, s)
        ψ1d = hermite_timestep_hardcoded(prob.co_dynamic, Vector(prob.ψ0), 0.0, Δt, wpd)
        @test maximum(abs.(ψ1d .- ref[:, end])) < 1e-11
    end
end

# =============================================================================
# 6. Saving options: history shape, save_every subsampling, final-only
# =============================================================================
@testset "Saving options" begin
    prob = hermite_variable_problem()
    nsteps = 100
    Δt = T / nsteps

    full = hermite_solve_hardcoded(prob.co_static, prob.ψ0, Δt, nsteps, 2)
    @test size(full) == (2, nsteps + 1)

    sub = hermite_solve_hardcoded(prob.co_static, prob.ψ0, Δt, nsteps, 2; save_every = 10)
    @test size(sub) == (2, length(0:10:nsteps))
    @test sub[:, 1] ≈ full[:, 1]
    @test sub[:, end] ≈ full[:, end]
    @test sub[:, 2] ≈ full[:, 11]

    fin = hermite_solve_hardcoded(prob.co_static, prob.ψ0, Δt, nsteps, 2; save_final_only = true)
    @test fin isa AbstractVector
    @test fin ≈ full[:, end]

    fin_d = hermite_solve_hardcoded(prob.co_dynamic, prob.ψ0, Δt, nsteps, 2; save_final_only = true)
    @test maximum(abs.(fin_d .- fin)) < 1e-11
end

# =============================================================================
# 7. Allocation: the static step and the dynamic matvec are allocation-free.
# =============================================================================
@testset "Allocation" begin
    prob = hermite_variable_problem()
    Δt = 0.01
    for s in 0:2
        vs = Val(s)
        wp = _static_hermite_weights(vs, Δt)
        ψ = SVector{2,ComplexF64}(prob.ψ0)
        An = _matderivs(prob.co_static, 0.0, vs)
        Anp1 = _matderivs(prob.co_static, Δt, vs)
        step() = _hermite_step_static(An, Anp1, ψ, wp)
        step()
        @test (@allocated step()) == 0

        wpd = _dynamic_hermite_weights(vs, Δt)
        ws = _FilonDynWS(2)
        x = Vector(prob.ψ0); out = zeros(ComplexF64, 2)
        ops = _opderivs(prob.co_dynamic, 0.0, vs)
        applyM() = _apply_M_hermite!(out, x, ops, wpd.WE, ws, vs)
        applyM()
        @test (@allocated applyM()) == 0
    end
end

# =============================================================================
# 8. Argument validation
# =============================================================================
@testset "Argument validation" begin
    prob = hermite_variable_problem()
    @test_throws ArgumentError hermite_solve_hardcoded(prob.co_static, prob.ψ0, 0.1, 10, 3)
    @test_throws ArgumentError hermite_solve_hardcoded(prob.co_static, prob.ψ0, 0.1, 0, 1)
    @test_throws ArgumentError hermite_solve_hardcoded(prob.co_static, prob.ψ0, 0.1, 10, 1; save_every = 0)
    @test_throws DimensionMismatch hermite_solve_hardcoded(prob.co_static, [1.0 + 0im], 0.1, 10, 1)
end
