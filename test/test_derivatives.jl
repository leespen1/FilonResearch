using FilonResearch, Test, Polynomials, LinearAlgebra
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
        @test multiple_general_leibniz_rule(one_derivs, cos_derivs) == cos_derivs
        rand_derivs = rand(MersenneTwister(0), 4)
        @test multiple_general_leibniz_rule(rand_derivs, one_derivs) == rand_derivs
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

        @test multiple_general_leibniz_rule(p1_derivs, p2_derivs) ≈ pprod_derivs rtol=1e-14
    end

    @testset "Exponential case: exp(αx)×exp(βx), x=0" begin
        α = 1.2
        β = -0.7
        n_derivs = 5

        exp1_derivs = ComplexF64[(α^m) for m in 0:n_derivs]
        exp2_derivs = ComplexF64[(β^m) for m in 0:n_derivs]
        expprod_derivs = multiple_general_leibniz_rule(exp1_derivs, exp2_derivs)
        expprod_derivs_exact = ComplexF64[((α + β)^m) for m in 0:n_derivs]
        @test expprod_derivs ≈ expprod_derivs_exact atol=1e-14
    end

end

@testset "Linear ODE Derivatives" begin
    @testset "Scalar case" begin
        u = 4.0
        @testset "t=$t" for t in (0, pi/3, pi/4)
            A = cos(2*t)
            dA = -2*sin(2*t)
            d2A = -4*cos(2*t)
            A_derivs = [A, dA, d2A]

            du = cos(2*t)*u
            d2u = (cos(2*t)^2 - 2*sin(2*t))*u
            d3u = (cos(2*t)^3 - 6*cos(2*t)*sin(2*t) - 4*cos(2*t))*u
            u_derivs_true = [u, du, d2u, d3u]

            u_derivs_computed = linear_ode_derivs(A_derivs, u)
            @test u_derivs_computed ≈ u_derivs_true atol=1e-14 rtol=1e-14
        end
    end

    @testset "Agreement between soft and hardcoded versions" begin
        N = 4
        u = rand(MersenneTwister(0), ComplexF64, N)
        A_derivs = [rand(MersenneTwister(i), ComplexF64, N, N) for i in 0:2]

        u_derivs_soft = linear_ode_derivs(A_derivs, u)
        u_derivs_hard = linear_ode_derivs_hardcoded(A_derivs, u)
        for derivative_order in 0:3
            @test u_derivs_soft[1+derivative_order] ≈ u_derivs_hard[1+derivative_order] atol=1e-14 rtol=1e-14
        end
    end

end
