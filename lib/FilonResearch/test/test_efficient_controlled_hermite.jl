using FilonResearch, Test, LinearAlgebra
using StaticArrays

# =============================================================================
# Efficient controlled Hermite method — the ω = 0 case of the efficient
# controlled Filon method.  The quadrature is split over each control matrix and
# the derivatives are moved onto the scalar controls; no frequency is considered.
# It is algebraically identical to `hermite_solve_hardcoded` over the full A(t),
# so the two must agree to round-off — only the matvec ordering differs.
#
# Test problem: a drift plus two control matrices with smooth scalar controls,
#   dψ/dt = [ A0 + c1(t) A1 + c2(t) A2 ] ψ.
# =============================================================================

# Polynomial control with analytic derivatives (degree 2; higher orders zero).
mkpoly(a, b, c) = FunctionControl{ComplexF64}(
    (t, n) -> n == 0 ? ComplexF64(a + b*t + c*t^2) :
              (n == 1 ? ComplexF64(b + 2c*t) : (n == 2 ? ComplexF64(2c) : 0.0 + 0im)))

function hermite_problem()
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]
    A2 = ComplexF64[0 0 -im; 0 0 0; -im 0 0]
    c1 = mkpoly(0.3, 0.1, -0.05)
    c2 = CarrierControl(mkpoly(0.2, -0.04, 0.03), 1.7)   # carrier folded in; treated as a scalar
    co = ControlledOperator((ConstantControl(1.0 + 0im), c1, c2), [A0, A1, A2])
    return (; co, ψ0 = ComplexF64[1, 0, 0])
end

const EHT = 1.0

# =============================================================================
# 1. Primary correctness: efficient controlled Hermite == regular Hermite
# =============================================================================
@testset "Matches hermite_solve_hardcoded" begin
    prob = hermite_problem()
    for s in 0:2, ns in (20, 50, 100)
        ref = hermite_solve_hardcoded(prob.co, prob.ψ0, EHT/ns, ns, s;
                                      save_final_only=true, variant=:dynamic)
        eff = efficient_controlled_hermite_solve(prob.co, prob.ψ0, EHT/ns, ns, s;
                                                 save_final_only=true)
        @test maximum(abs.(eff .- ref)) < 1e-11
    end
end

# =============================================================================
# 2. Convergence orders 2 / 4 / 6
# =============================================================================
@testset "Convergence orders" begin
    prob = hermite_problem()
    uref = hermite_solve_hardcoded(prob.co, prob.ψ0, EHT/20000, 20000, 2;
                                   save_final_only=true, variant=:dynamic)
    for s in 0:2
        errs = Float64[]
        for ns in (10, 20, 40, 80)
            u = efficient_controlled_hermite_solve(prob.co, prob.ψ0, EHT/ns, ns, s;
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
# 3. Fewer dense matvecs than regular Hermite, independent of control structure
# =============================================================================
mutable struct _CountMatH{M<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    A::M
    count::Base.RefValue{Int}
end
Base.size(c::_CountMatH) = size(c.A)
Base.getindex(c::_CountMatH, i...) = getindex(c.A, i...)
function LinearAlgebra.mul!(y::AbstractVector, c::_CountMatH, x::AbstractVector)
    c.count[] += 1
    return mul!(y, c.A, x)
end
function LinearAlgebra.mul!(y::AbstractVector, c::_CountMatH, x::AbstractVector, α::Number, β::Number)
    c.count[] += 1
    return mul!(y, c.A, x, α, β)
end

@testset "Fewer matvecs than regular Hermite" begin
    E = [1.0, 2.0, 3.5]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im 0; -im 0 -im; 0 -im 0]
    ψ0 = ComplexF64[1, 0, 0]
    c1 = mkpoly(0.3, 0.1, -0.05)
    for s in 1:2
        ctr_r = Ref(0)
        co_r = ControlledOperator((ConstantControl(1.0 + 0im), c1),
                                  [_CountMatH(A0, ctr_r), _CountMatH(A1, ctr_r)])
        hermite_solve_hardcoded(co_r, ψ0, EHT/8, 8, s; save_final_only=true, variant=:dynamic)
        ctr_e = Ref(0)
        co_e = ControlledOperator((ConstantControl(1.0 + 0im), c1),
                                  [_CountMatH(A0, ctr_e), _CountMatH(A1, ctr_e)])
        efficient_controlled_hermite_solve(co_e, ψ0, EHT/8, 8, s; save_final_only=true)
        @test ctr_e[] < ctr_r[]
    end
end

# A full FunctionControl pulse and an equivalent SumControl of carriers realize the
# same A(t); the matvec count is the same either way (derivatives are on scalars).
@testset "Matvec count independent of control representation" begin
    E = [1.0, 2.0]
    A0 = ComplexF64(-im) .* diagm(E)
    A1 = ComplexF64[0 -im; -im 0]
    ψ0 = ComplexF64[1, 0]
    e1 = mkpoly(0.2, 0.1, 0.0); e2 = mkpoly(-0.1, 0.05, 0.0)
    ν = 0.9
    # full pulse c(t) = e1(t) e^{iνt} + e2(t) e^{-iνt}, as one FunctionControl ...
    # (`derivative` is qualified: `using Polynomials` in a sibling test file shadows
    # the bare name in the shared runtests scope.)
    cp = CarrierControl(e1, ν); cm = CarrierControl(e2, -ν)
    cfull = FunctionControl{ComplexF64}((t, n) ->
        FilonResearch.derivative(cp, t, Derivative(n)) +
        FilonResearch.derivative(cm, t, Derivative(n)))
    # ... and as a SumControl of the two carriers (same A(t))
    csum = SumControl(CarrierControl(e1, ν), CarrierControl(e2, -ν))

    ctr_f = Ref(0)
    co_f = ControlledOperator((ConstantControl(1.0 + 0im), cfull),
                              [_CountMatH(A0, ctr_f), _CountMatH(A1, ctr_f)])
    res_f = efficient_controlled_hermite_solve(co_f, ψ0, EHT/16, 16, 2; save_final_only=true)
    ctr_s = Ref(0)
    co_s = ControlledOperator((ConstantControl(1.0 + 0im), csum),
                              [_CountMatH(A0, ctr_s), _CountMatH(A1, ctr_s)])
    res_s = efficient_controlled_hermite_solve(co_s, ψ0, EHT/16, 16, 2; save_final_only=true)

    @test ctr_f[] == ctr_s[]
    @test maximum(abs.(res_f .- res_s)) < 1e-12
end

# =============================================================================
# 4. Saving options and argument validation
# =============================================================================
@testset "Hermite: saving options" begin
    prob = hermite_problem()
    nsteps = 100; Δt = EHT / nsteps
    full = efficient_controlled_hermite_solve(prob.co, prob.ψ0, Δt, nsteps, 2)
    @test size(full) == (3, nsteps + 1)
    sub = efficient_controlled_hermite_solve(prob.co, prob.ψ0, Δt, nsteps, 2; save_every=10)
    @test size(sub) == (3, length(0:10:nsteps))
    @test sub[:, end] ≈ full[:, end]
    @test sub[:, 2] ≈ full[:, 11]
    fin = efficient_controlled_hermite_solve(prob.co, prob.ψ0, Δt, nsteps, 2; save_final_only=true)
    @test fin isa AbstractVector
    @test fin ≈ full[:, end]
end

@testset "Hermite: argument validation" begin
    prob = hermite_problem()
    @test_throws ArgumentError efficient_controlled_hermite_solve(prob.co, prob.ψ0, 0.1, 10, 3)
    @test_throws ArgumentError efficient_controlled_hermite_solve(prob.co, prob.ψ0, 0.1, 0, 1)
    @test_throws DimensionMismatch efficient_controlled_hermite_solve(prob.co, ComplexF64[1, 0], 0.1, 10, 1)
end
