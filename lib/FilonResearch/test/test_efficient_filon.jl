using FilonResearch, Test, LinearAlgebra
using StaticArrays

# =============================================================================
# Efficient Filon method: the A_k-factored (Appendix B) reorganization of regular
# Filon WITHOUT carrier resolution.  It is algebraically identical to
# `filon_solve_hardcoded` on the same operator — only the matrix-vector ordering
# differs — but applies each distinct matrix only s+1 times per step (not
# (s+1)(s+2)/2).  Carriers are folded into the control envelope and differentiated
# as generic scalars; the ansatz frequencies ω enter only through the diagonal
# weights.
#
# Test problem: a drift plus TWO control matrices, each driven by a single
# carrier-bearing FourierControl envelope.
# =============================================================================

function filon_problem()
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]              # -i (tridiagonal)
    A2 = ComplexF64[0 0 -im; 0 0 0; -im 0 0]
    c1 = FourierControl(0.2, [0.5], [0.3], 0.7)
    c2 = FourierControl(0.15, [-0.25], [0.1], 1.3)
    co = ControlledOperator((ConstantControl(1.0 + 0im), c1, c2), [A0, A1, A2])
    return (; co, ψ0 = ComplexF64[1, 0, 0], ωans = -E)
end

const EFT = 1.0

# =============================================================================
# 1. Primary correctness: identical to filon_solve_hardcoded (to round-off)
# =============================================================================
@testset "Efficient Filon matches regular Filon" begin
    prob = filon_problem()
    for s in 0:2, ns in (20, 50, 100)
        ref = filon_solve_hardcoded(prob.co, prob.ψ0, prob.ωans, EFT/ns, ns, s;
                                    save_final_only=true, variant=:dynamic)
        eff = efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, EFT/ns, ns, s;
                                    save_final_only=true)
        @test maximum(abs.(eff .- ref)) < 1e-11
    end
end

# =============================================================================
# 2. Convergence orders 2 / 4 / 6
# =============================================================================
@testset "Convergence orders" begin
    prob = filon_problem()
    Af(t)   = materialize(evaluate(prob.co, t, Derivative{0}()))
    dAf(t)  = materialize(evaluate(prob.co, t, Derivative{1}()))
    ddAf(t) = materialize(evaluate(prob.co, t, Derivative{2}()))
    uref = filon_solve((Af, dAf, ddAf), prob.ψ0, prob.ωans, EFT, 20000, 2)[:, end]
    for s in 0:2
        errs = Float64[]
        for ns in (10, 20, 40, 80)
            u = efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, EFT/ns, ns, s;
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
# 3. Dense matvec count is (s+1)·nHam per application, and independent of how
#    the control is expressed (the efficiency claim)
# =============================================================================
# A matrix wrapper that counts mul! calls, so we can verify how often each
# distinct control matrix is applied per step.
mutable struct _CountMatF{M<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    A::M
    count::Base.RefValue{Int}
end
Base.size(c::_CountMatF) = size(c.A)
Base.getindex(c::_CountMatF, i...) = getindex(c.A, i...)
function LinearAlgebra.mul!(y::AbstractVector, c::_CountMatF, x::AbstractVector)
    c.count[] += 1
    return mul!(y, c.A, x)
end
function LinearAlgebra.mul!(y::AbstractVector, c::_CountMatF, x::AbstractVector, α::Number, β::Number)
    c.count[] += 1
    return mul!(y, c.A, x, α, β)
end

@testset "Dense matvec count = (s+1)·nHam" begin
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]
    ψ0 = ComplexF64[1, 0, 0]; ωans = -E
    nHam = 2

    # The dense-matvec count per application of S^s is (s+1)·nHam.  Over one step
    # the apply runs once on the explicit side and `niters` times inside the
    # implicit GMRES solve, so total = (1 + niters)·(s+1)·nHam.
    for s in 0:2
        ctr = Ref(0)
        mats = [_CountMatF(A0, ctr), _CountMatF(A1, ctr)]
        co = ControlledOperator((ConstantControl(1.0 + 0im), FourierControl(0.1, [0.05], [0.02], 0.4)), mats)
        st = FilonSolveStats()
        efficient_filon_solve(co, ψ0, ωans, EFT/8, 1, s; save_final_only=true, stats=st)
        per_apply = ctr[] / (1 + st.gmres_niters[1])
        @test per_apply == (s + 1) * nHam
    end

    # Independent of how the carrier is expressed: folding it into one
    # FourierControl, or splitting it across a SumControl of carriers, realizes
    # the same A(t) and so costs the same dense matvecs.
    mkenv(j) = FourierControl(0.1j, [0.05j], [0.02j], 0.3 + 0.1j)
    counts = Int[]
    for nfreq in (1, 3)
        ctr = Ref(0)
        mats = [_CountMatF(A0, ctr), _CountMatF(A1, ctr)]
        carriers = SumControl(ntuple(j -> CarrierControl(mkenv(j), 0.4j), nfreq)...)
        co = ControlledOperator((ConstantControl(1.0 + 0im), carriers), mats)
        efficient_filon_solve(co, ψ0, ωans, EFT/8, 8, 2; save_final_only=true)
        push!(counts, ctr[])
    end
    @test counts[1] == counts[2]

    # And strictly fewer dense matvecs than regular Filon (which applies each A_k
    # (s+1)(s+2)/2 times) on the same operator, for s ≥ 1.
    for s in 1:2
        ce = Ref(0); cr = Ref(0)
        co_e = ControlledOperator((ConstantControl(1.0 + 0im), FourierControl(0.1, [0.05], [0.02], 0.4)),
                                  [_CountMatF(A0, ce), _CountMatF(A1, ce)])
        co_r = ControlledOperator((ConstantControl(1.0 + 0im), FourierControl(0.1, [0.05], [0.02], 0.4)),
                                  [_CountMatF(A0, cr), _CountMatF(A1, cr)])
        efficient_filon_solve(co_e, ψ0, ωans, EFT/8, 8, s; save_final_only=true)
        filon_solve_hardcoded(co_r, ψ0, ωans, EFT/8, 8, s; save_final_only=true, variant=:dynamic)
        @test ce[] < cr[]
    end
end

# =============================================================================
# 4. Saving options and argument validation
# =============================================================================
@testset "Efficient Filon: saving options" begin
    prob = filon_problem()
    nsteps = 100; Δt = EFT / nsteps
    full = efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, Δt, nsteps, 2)
    @test size(full) == (3, nsteps + 1)
    sub = efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, Δt, nsteps, 2; save_every=10)
    @test size(sub) == (3, length(0:10:nsteps))
    @test sub[:, end] ≈ full[:, end]
    @test sub[:, 2] ≈ full[:, 11]
    fin = efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, Δt, nsteps, 2; save_final_only=true)
    @test fin isa AbstractVector
    @test fin ≈ full[:, end]
end

@testset "Efficient Filon: argument validation" begin
    prob = filon_problem()
    @test_throws ArgumentError efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, 0.1, 10, 3)
    @test_throws ArgumentError efficient_filon_solve(prob.co, prob.ψ0, prob.ωans, 0.1, 0, 1)
    @test_throws DimensionMismatch efficient_filon_solve(prob.co, prob.ψ0, [1.0], 0.1, 10, 1)
    @test_throws DimensionMismatch efficient_filon_solve(prob.co, ComplexF64[1, 0], prob.ωans, 0.1, 10, 1)
end
