using Test, FilonResearch, Polynomials
using Random: MersenneTwister

@testset "Hard-coded Hermite cardinal polynomial correctness" begin
    @testset "s=$s" for s=0:3
        polynomials = hermite_cardinal_polynomials(s)
        @testset "ℓ_1$(j-1)" for (j, poly) in enumerate(polynomials)
            @testset "Derivative $k" for k in 0:s
                if k == j-1
                    @test derivative(poly, k)(1) ≈ 1 atol=1e-15
                else
                    @test derivative(poly, k)(1) ≈ 0 atol=1e-15
                end
                @test derivative(poly, k)(-1) ≈ 0 atol=1e-15
            end
        end
    end
end

@testset "Correct Polynomial Derivative Matrix" begin
    x = 1
    n_deriv = 4
    degree = 3
    polynomial_derivative_matrix = derivative_monomial_matrix(-1, n_deriv, degree)
    hard_coded_matrix = [1 -1  1 -1
                         0  1 -2  3
                         0  0  2 -6
                         0  0  0  6
                         0  0  0  0]
    @test polynomial_derivative_matrix == hard_coded_matrix
end

@testset "Linear-solve Hermite interpolating polynomial correctness" begin
    @testset "Reproduces hard-coded cardinal polynomials" begin
        a = -1
        b = 1
        @testset "s=$s" for s=0:3
            hard_coded_polys = hermite_cardinal_polynomials(s)
            @testset "ℓ_1$k" for k in 0:s

                fa_derivs = zeros(1+s)
                fb_derivs = [(deriv_order == k) ? 1 : 0 for deriv_order in 0:s]
                linear_solve_poly = hermite_interpolating_polynomial(a, b, fa_derivs, fb_derivs) 
                hard_coded_poly = hard_coded_polys[1+k]
                for i in eachindex(linear_solve_poly)
                    @test linear_solve_poly[i] ≈ hard_coded_poly[i] atol=1e-14 rtol=1e-14
                end
            end
        end
    end

    @testset "Random Values" begin
        # WARNING: the condition number of the LHS gets worse as number of
        # derivatives increases, or as a and b get closer together.  
        a = -1.5
        b = 2
        fa_derivs = [0.09, 0.7, 0.5]
        fb_derivs = [0.88, 0.26, 0.66, 0.35]

        ## With these values, test fails due to poor conditioning
        #a, b = rand(MersenneTwister(0), 2)
        #fa_derivs = rand(MersenneTwister(1), 5)
        #fb_derivs = rand(MersenneTwister(2), 7)
        
        poly = hermite_interpolating_polynomial(a, b, fa_derivs, fb_derivs)
        @show a b fa_derivs fb_derivs poly
        for i in 1:length(fa_derivs)
            deriv_order = i-1
            @test derivative(poly, deriv_order)(a) ≈ fa_derivs[i] atol=1e-15 rtol=1e-15
        end
        for i in 1:length(fa_derivs)
            deriv_order = i-1
            @test derivative(poly, deriv_order)(b) ≈ fb_derivs[i] atol=1e-15 rtol=1e-15
        end
    end
end
