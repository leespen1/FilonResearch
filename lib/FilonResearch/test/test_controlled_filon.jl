using FilonResearch, Test, LinearAlgebra
using StaticArrays

# =============================================================================
# Controlled Filon method (Appendix B).  Test problems use a drift plus a single
# carrier-wave control, in matched static (SMatrix) and dynamic (Vector) layouts.
#   dψ/dt = [ A0  +  c̃(t) e^{i ωc t} A1 ] ψ,    A0 = -i diag(E),  A1 = -i σx.
# =============================================================================

function carrier_problem(; ωc = 1.3)
    E = [1.0, 2.0]
    A0 = ComplexF64[-im*E[1] 0; 0 -im*E[2]]
    A1 = ComplexF64[0 -im; -im 0]                     # -i σx
    env = FourierControl(0.2, [0.5], [0.3], 0.7)      # slow envelope c̃(t)
    ctrls = (ConstantControl(1.0), CarrierControl(env, ωc))
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(A0), SMatrix{2,2}(A1)))
    co_dynamic = ControlledOperator(ctrls, [A0, A1])
    return (; co_static, co_dynamic, ψ0 = ComplexF64[1.0, 0.0],
              ωans = [-1.0, -2.0])                    # ansatz freqs (= -E)
end

# Independent reference: regular filon_solve on the full (carrier-bearing) A(t)
# at a fine grid.
function carrier_reference(co, ψ0, ωans, T)
    Af(t)   = materialize(evaluate(co, t, Derivative{0}()))
    dAf(t)  = materialize(evaluate(co, t, Derivative{1}()))
    ddAf(t) = materialize(evaluate(co, t, Derivative{2}()))
    return filon_solve((Af, dAf, ddAf), ψ0, ωans, T, 20000, 2)[:, end]
end

function cf_check_order(errors, expected)
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    # The steepest slope is the asymptotic rate; later refinements flatten once
    # they reach the (~1e-12) accuracy floor of the fine-grid reference.
    @test maximum(orders) > expected - 0.5
    @test errors[end] < errors[1]
end

const CFT = 1.0
const CF_NS = [10, 20, 40, 80]

# =============================================================================
# 1. With zero carriers, controlled Filon == regular Filon (exactly)
# =============================================================================
@testset "Reduces to regular Filon at zero carrier" begin
    E = [1.0, 2.0]
    A0 = ComplexF64[-im*E[1] 0; 0 -im*E[2]]
    A1 = ComplexF64[0 -im; -im 0]
    env = FourierControl(0.2, [0.5], [0.3], 0.7)
    ψ0 = ComplexF64[1.0, 0.0]; ωans = [-1.0, -2.0]
    # ordinary controls (carrier_frequency = 0) and an explicit zero CarrierControl
    for ctrls in (((ConstantControl(1.0), env)),
                  ((ConstantControl(1.0), CarrierControl(env, 0.0))))
        co_s = ControlledOperator(ctrls, (SMatrix{2,2}(A0), SMatrix{2,2}(A1)))
        co_d = ControlledOperator(ctrls, [A0, A1])
        for s in 0:2
            cf_s = controlled_filon_solve(co_s, ψ0, ωans, CFT/20, 20, s; save_final_only=true)
            cf_d = controlled_filon_solve(co_d, ψ0, ωans, CFT/20, 20, s; save_final_only=true)
            reg  = filon_solve_hardcoded(co_s, ψ0, ωans, CFT/20, 20, s; save_final_only=true)
            @test maximum(abs.(cf_s .- reg)) < 1e-12
            @test maximum(abs.(cf_d .- reg)) < 1e-11
        end
    end
end

# =============================================================================
# 2. Convergence orders 2/4/6 on a carrier-wave problem, both variants
# =============================================================================
@testset "Carrier-wave convergence orders" begin
    prob = carrier_problem()
    uref = carrier_reference(prob.co_dynamic, prob.ψ0, prob.ωans, CFT)
    for s in 0:2, (lbl, co) in (("static", prob.co_static), ("dynamic", prob.co_dynamic))
        errs = Float64[]
        for ns in CF_NS
            u = controlled_filon_solve(co, prob.ψ0, prob.ωans, CFT/ns, ns, s; save_final_only=true)
            push!(errs, maximum(abs.(u .- uref)))
        end
        @testset "s=$s ($lbl)" begin
            cf_check_order(errs, 2s + 2.0)
        end
    end
end

# =============================================================================
# 3. Static and dynamic variants agree
# =============================================================================
@testset "Controlled: static vs dynamic agreement" begin
    prob = carrier_problem()
    ns = 50
    for s in 0:2
        us = controlled_filon_solve(prob.co_static,  prob.ψ0, prob.ωans, CFT/ns, ns, s; save_final_only=true)
        ud = controlled_filon_solve(prob.co_dynamic, prob.ψ0, prob.ωans, CFT/ns, ns, s; save_final_only=true)
        @test maximum(abs.(us .- ud)) < 1e-11
    end
end

# =============================================================================
# 4. Saving options
# =============================================================================
@testset "Controlled: saving options" begin
    prob = carrier_problem()
    nsteps = 100; Δt = CFT / nsteps
    full = controlled_filon_solve(prob.co_static, prob.ψ0, prob.ωans, Δt, nsteps, 2)
    @test size(full) == (2, nsteps + 1)
    sub = controlled_filon_solve(prob.co_static, prob.ψ0, prob.ωans, Δt, nsteps, 2; save_every=10)
    @test size(sub) == (2, length(0:10:nsteps))
    @test sub[:, end] ≈ full[:, end]
    @test sub[:, 2] ≈ full[:, 11]
    fin = controlled_filon_solve(prob.co_dynamic, prob.ψ0, prob.ωans, Δt, nsteps, 2; save_final_only=true)
    @test fin isa AbstractVector
    @test fin ≈ full[:, end]
end

# =============================================================================
# 5. Argument validation
# =============================================================================
@testset "Controlled: argument validation" begin
    prob = carrier_problem()
    @test_throws ArgumentError controlled_filon_solve(prob.co_static, prob.ψ0, prob.ωans, 0.1, 10, 3)
    @test_throws ArgumentError controlled_filon_solve(prob.co_static, prob.ψ0, prob.ωans, 0.1, 0, 1)
    @test_throws DimensionMismatch controlled_filon_solve(prob.co_static, prob.ψ0, [1.0], 0.1, 10, 1)
end
