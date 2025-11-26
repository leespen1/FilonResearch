using Test, FilonResearch, Polynomials

@testset "Hermite Cardinal Polynomial Correctness for s=$s" for s=0:3
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
