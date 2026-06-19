using FilonResearch, Test, LinearAlgebra
using StaticArrays

# =============================================================================
# Efficient (generator-form) controlled Filon method (Appendix B).  The method
# is identical to `controlled_filon_solve` but gathers every carrier sharing a
# control matrix into one `SumControl`, so each distinct matrix is applied once
# per step regardless of the number of carriers.
#
# Test problem: a drift plus TWO control matrices, each driven by TWO carriers.
#   dψ/dt = [ A0 + (Σ_l c̃_{1,l} e^{i ν_{1,l} t}) A1 + (Σ_l c̃_{2,l} e^{i ν_{2,l} t}) A2 ] ψ
# Built in a grouped form (SumControl per matrix) and an equivalent per-carrier
# form (one duplicated matrix per carrier), which `controlled_filon_solve` uses.
# =============================================================================

function multicarrier_problem()
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]              # -i (tridiagonal)
    A2 = ComplexF64[0 0 -im; 0 0 0; -im 0 0]
    e1a = FourierControl(0.2, [0.5], [0.3], 0.7)
    e1b = FourierControl(-0.1, [0.2], [-0.15], 1.1)
    e2a = FourierControl(0.15, [-0.25], [0.1], 0.5)
    e2b = FourierControl(0.05, [0.1], [0.2], 1.3)
    ν1 = (0.8, -1.3); ν2 = (1.7, 0.5)

    # Grouped (efficient): one matrix per A_k, a SumControl over its carriers.
    co_grouped = ControlledOperator(
        (ConstantControl(1.0 + 0im),
         SumControl(CarrierControl(e1a, ν1[1]), CarrierControl(e1b, ν1[2])),
         SumControl(CarrierControl(e2a, ν2[1]), CarrierControl(e2b, ν2[2]))),
        [A0, A1, A2])
    # Per-carrier reference: one duplicated matrix per carrier term.
    co_percarrier = ControlledOperator(
        (ConstantControl(1.0 + 0im),
         CarrierControl(e1a, ν1[1]), CarrierControl(e1b, ν1[2]),
         CarrierControl(e2a, ν2[1]), CarrierControl(e2b, ν2[2])),
        [A0, A1, A1, A2, A2])
    return (; co_grouped, co_percarrier, ψ0 = ComplexF64[1, 0, 0], ωans = -E)
end

const EFT = 1.0

# =============================================================================
# 1. The grouped and per-carrier representations realize the same A(t)
# =============================================================================
@testset "Grouped operator reproduces the per-carrier A(t)" begin
    prob = multicarrier_problem()
    for t in (0.0, 0.31, 0.77, 1.4), d in (Derivative{0}(), Derivative{1}(), Derivative{2}())
        Ag = materialize(evaluate(prob.co_grouped, t, d))
        Ap = materialize(evaluate(prob.co_percarrier, t, d))
        @test maximum(abs.(Ag .- Ap)) < 1e-13
    end
end

# =============================================================================
# 2. Primary correctness: efficient solve == existing controlled_filon_solve
# =============================================================================
@testset "Efficient matches controlled_filon_solve" begin
    prob = multicarrier_problem()
    for s in 0:2, ns in (20, 50, 100)
        ref = controlled_filon_solve(prob.co_percarrier, prob.ψ0, prob.ωans, EFT/ns, ns, s;
                                     save_final_only=true, variant=:dynamic)
        eff = efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, EFT/ns, ns, s;
                                               save_final_only=true)
        @test maximum(abs.(eff .- ref)) < 1e-11
    end
end

# =============================================================================
# 3. With zero carriers, reduces to the regular Filon method
# =============================================================================
@testset "Reduces to regular Filon at zero carrier" begin
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]
    ea = FourierControl(0.2, [0.5], [0.3], 0.7)
    eb = FourierControl(-0.1, [0.2], [-0.15], 1.1)
    ψ0 = ComplexF64[1, 0, 0]; ωans = -E
    co_grouped = ControlledOperator(
        (ConstantControl(1.0 + 0im), SumControl(CarrierControl(ea, 0.0), CarrierControl(eb, 0.0))),
        [A0, A1])
    co_filon = ControlledOperator((ConstantControl(1.0 + 0im), SumControl(ea, eb)), [A0, A1])
    for s in 0:2
        eff = efficient_controlled_filon_solve(co_grouped, ψ0, ωans, EFT/30, 30, s; save_final_only=true)
        reg = filon_solve_hardcoded(co_filon, ψ0, ωans, EFT/30, 30, s; save_final_only=true, variant=:dynamic)
        @test maximum(abs.(eff .- reg)) < 1e-11
    end
end

# =============================================================================
# 4. Convergence orders 2 / 4 / 6 on the multi-carrier problem
# =============================================================================
@testset "Multi-carrier convergence orders" begin
    prob = multicarrier_problem()
    # Reference: a fine-grid regular Filon solve on the full carrier-bearing 𝒜(t).
    Af(t)   = materialize(evaluate(prob.co_grouped, t, Derivative{0}()))
    dAf(t)  = materialize(evaluate(prob.co_grouped, t, Derivative{1}()))
    ddAf(t) = materialize(evaluate(prob.co_grouped, t, Derivative{2}()))
    uref = filon_solve((Af, dAf, ddAf), prob.ψ0, prob.ωans, EFT, 20000, 2)[:, end]
    for s in 0:2
        errs = Float64[]
        for ns in (10, 20, 40, 80)
            u = efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, EFT/ns, ns, s;
                                                 save_final_only=true)
            push!(errs, maximum(abs.(u .- uref)))
        end
        orders = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
        @testset "s=$s" begin
            @test maximum(orders) > 2s + 2 - 0.5
            @test errs[end] < errs[1]
        end
    end
end

# =============================================================================
# 5. Dense matvec count is independent of the number of carriers (the point)
# =============================================================================
# A matrix wrapper that counts mul! calls, so we can verify how often each
# distinct control matrix is applied per step.
mutable struct _CountMat{M<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    A::M
    count::Base.RefValue{Int}
end
Base.size(c::_CountMat) = size(c.A)
Base.getindex(c::_CountMat, i...) = getindex(c.A, i...)
function LinearAlgebra.mul!(y::AbstractVector, c::_CountMat, x::AbstractVector)
    c.count[] += 1
    return mul!(y, c.A, x)
end
function LinearAlgebra.mul!(y::AbstractVector, c::_CountMat, x::AbstractVector, α::Number, β::Number)
    c.count[] += 1
    return mul!(y, c.A, x, α, β)
end

@testset "Dense matvecs independent of carrier count" begin
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]
    ψ0 = ComplexF64[1, 0, 0]; ωans = -E
    mkenv(j) = FourierControl(0.1j, [0.05j], [0.02j], 0.3 + 0.1j)

    counts = Int[]
    for nfreq in (1, 3)
        ctr = Ref(0)
        mats = [_CountMat(A0, ctr), _CountMat(A1, ctr)]
        carriers = SumControl(ntuple(j -> CarrierControl(mkenv(j), 0.4j), nfreq)...)
        co = ControlledOperator((ConstantControl(1.0 + 0im), carriers), mats)
        efficient_controlled_filon_solve(co, ψ0, ωans, EFT/8, 8, 2; save_final_only=true)
        push!(counts, ctr[])
    end
    # Same matrix count (1 drift + 1 control) ⇒ identical dense matvecs whether
    # the control has 1 or 3 carriers.
    @test counts[1] == counts[2]
end

# =============================================================================
# 6. Saving options and argument validation
# =============================================================================
@testset "Efficient: saving options" begin
    prob = multicarrier_problem()
    nsteps = 100; Δt = EFT / nsteps
    full = efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, Δt, nsteps, 2)
    @test size(full) == (3, nsteps + 1)
    sub = efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, Δt, nsteps, 2; save_every=10)
    @test size(sub) == (3, length(0:10:nsteps))
    @test sub[:, end] ≈ full[:, end]
    @test sub[:, 2] ≈ full[:, 11]
    fin = efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, Δt, nsteps, 2; save_final_only=true)
    @test fin isa AbstractVector
    @test fin ≈ full[:, end]
end

@testset "Efficient: argument validation" begin
    prob = multicarrier_problem()
    @test_throws ArgumentError efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, 0.1, 10, 3)
    @test_throws ArgumentError efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, prob.ωans, 0.1, 0, 1)
    @test_throws DimensionMismatch efficient_controlled_filon_solve(prob.co_grouped, prob.ψ0, [1.0], 0.1, 10, 1)
    @test_throws DimensionMismatch efficient_controlled_filon_solve(prob.co_grouped, ComplexF64[1, 0], prob.ωans, 0.1, 10, 1)
end
