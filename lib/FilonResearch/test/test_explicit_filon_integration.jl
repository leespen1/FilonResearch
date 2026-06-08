using FilonResearch, Test, Polynomials
using SpecialFunctions: expint
@testset "Polynomial f(x)" begin
    f = Polynomial([1, 2, 3, 4]) # degree 3 polynomial
    indefinite_integral(x, ω) = -(1/ω^4)*im*exp(im*ω*x) * (
        ω*(ω*(c[1]*ω + c[2]*ω*x + im*c[2]) + c[3]*(ω^2*x^2 + 2im*ω*x - 2))
        + c[4]*(ω^3*x^3 + 3im*ω^2*x^2 -6*ω*x - 6im)
    )

    true_solution(a, b, ω) =  indefinite_integral(b, ω) - indefinite_integral(a, ω)
    a = -2
    b = 3

    fa_derivs = [derivative(f, m)(a) for m in 0:2]
    fb_derivs = [derivative(f, m)(b) for m in 0:2]

end

@testset "f(x) = x/(1+x^2)" begin
    # Warning: do not exceed s=5, or else the coefficeints in the rational
    # function f will become too big for Int64
    
    a = -1
    b = 1

    f = Polynomial([0, 1]) // Polynomial([1,0,1])
    Ei(x) = -expint(-x)
    indefinite_integral(x, ω) = 0.5*(exp(ω)*Ei(im*ω*(x+im)) + exp(-ω)*Ei(im*x*ω + ω))
    true_solution(a, b, ω) =  indefinite_integral(b, ω) - indefinite_integral(a, ω)

    @testset "a=-1, b=1" begin
        a = -1
        b = 1

        fa_derivs = [derivative(f, s)(a) for s in 0:5]
        fb_derivs = [derivative(f, s)(b) for s in 0:5]

        @testset "ω=20" begin
            ω=20
            true_sol = true_solution(a, b, ω)
            expected_errors = [1e-2, 1e-3, 1e-4, 1e-5, 1e-5, 1e-6]
            @testset "s=$s" for s in 0:5
                approx_sol = explicit_filon_integral(
                    ω, s, a, b, fa_derivs[1:1+s], fb_derivs[1:1+s]
                )
                @test approx_sol ≈ true_sol atol=expected_errors[1+s]
            end
        end
        @testset "ω=200" begin
            ω=200
            true_sol = true_solution(a, b, ω)
            expected_errors = [1e-4, 1e-6, 1e-8, 1e-10, 1e-11, 1e-13]
            @testset "s=$s" for s in 0:5
                approx_sol = explicit_filon_integral(
                    ω, s, a, b, fa_derivs[1:1+s], fb_derivs[1:1+s]
                )
                @test approx_sol ≈ true_sol atol=expected_errors[1+s]
            end
        end
    end

    @testset "a=-1.5, b=2" begin
        a = -1.5
        b = b=2

        fa_derivs = [derivative(f, s)(a) for s in 0:5]
        fb_derivs = [derivative(f, s)(b) for s in 0:5]

        @testset "ω=20" begin
            ω=20
            true_sol = true_solution(a, b, ω)
            expected_errors = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-6]
            @testset "s=$s" for s in 0:5
                approx_sol = explicit_filon_integral(
                    ω, s, a, b, fa_derivs[1:1+s], fb_derivs[1:1+s]
                )
                @test approx_sol ≈ true_sol atol=expected_errors[1+s]
            end
        end
        @testset "ω=200" begin
            ω=200
            true_sol = true_solution(a, b, ω)
            expected_errors = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-13]
            @testset "s=$s" for s in 0:5
                approx_sol = explicit_filon_integral(
                    ω, s, a, b, fa_derivs[1:1+s], fb_derivs[1:1+s]
                )
                @test approx_sol ≈ true_sol atol=expected_errors[1+s]
            end
        end
    end





    # @test explicit_filon_integral()
end
