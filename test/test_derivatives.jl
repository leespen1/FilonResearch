using FilonResearch, Test, Polynomials
import Random: MersenneTwister

@testset "Derivatives of e^(iωt)" begin
        n_derivs = 3
        @test exp_iωt_derivs(0, 5, n_derivs) == [1, 0, 0, 0]
        @test exp_iωt_derivs(1, 0, n_derivs) == [1, 1im, -1, -1im]
        @test exp_iωt_derivs(1, 2, n_derivs, pi_units=true) == [1, 1im, -1, -1im]
        @test exp_iωt_derivs(1, 1, n_derivs, pi_units=true) == [-1, -1im, 1, 1im]
        @test exp_iωt_derivs(2, 2, n_derivs, pi_units=true) == [1, 2im, -4, -8im]
        @test exp_iωt_derivs(1, 2pi, n_derivs) ≈ [1, 1im, -1, -1im] atol=1e-14
        @test exp_iωt_derivs(1, 1pi, n_derivs) ≈ [-1, -1im, 1, 1im] atol=1e-14
        @test exp_iωt_derivs(2, 2pi, n_derivs) ≈ [1, 2im, -4, -8im] atol=1e-14
end

@testset "General Leibniz Rule" begin
    @testset "Trivial case: f(x)×1" begin
        one_derivs = [1, 0, 0, 0]
        cos_derivs = [1, 0, -1, 0]
        @test general_leibniz_rule(one_derivs, cos_derivs) == cos_derivs
        rand_derivs = rand(MersenneTwister(0), 4)
        @test general_leibniz_rule(rand_derivs, one_derivs) == rand_derivs
    end

    @testset "Polynomial case: p₁(x)×p₂(x)" begin
        n_derivs = 3
        degree1 = 4
        degree2 = 5
        x = rand(MersenneTwister(0))
        p1 = Polynomial(rand(MersenneTwister(1), degree1))
        p2 = Polynomial(rand(MersenneTwister(2), degree2))
        p1_derivs = [derivative(p1, order)(x) for order in 0:n_derivs]
        p2_derivs = [derivative(p2, order)(x) for order in 0:n_derivs]

        pprod = p1*p2
        pprod_derivs = [derivative(pprod, order)(x) for order in 0:n_derivs]

        @test general_leibniz_rule(p1_derivs, p2_derivs) ≈ pprod_derivs rtol=1e-14
    end

    @testset "Exponential case: exp(αx)×exp(βx), x=0" begin
        α = 1.2
        β = -0.7
        n_derivs = 5

        exp1_derivs = ComplexF64[(α^m) for m in 0:n_derivs]
        exp2_derivs = ComplexF64[(β^m) for m in 0:n_derivs]
        expprod_derivs = general_leibniz_rule(exp1_derivs, exp2_derivs)
        expprod_derivs_exact = ComplexF64[((α + β)^m) for m in 0:n_derivs]
        @test expprod_derivs ≈ expprod_derivs_exact atol=1e-14
    end

end

