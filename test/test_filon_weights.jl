using FilonResearch, Test

@testset "Correct Filon moments" begin

    degree = 3
    ω = 10.0
    a = -1.5
    b = 2.3
    moments = filon_moments(degree, ω, a, b)

    moment_1_indefinite(x, ω) = -(im/ω) * exp(im*ω*x)
    moment_x_indefinite(x, ω) = exp(im*ω*x)*(1-im*ω*x) / ω^2
    moment_x²_indefinite(x, ω) = exp(im*ω*x)*(-im*ω^2*x^2 + 2*ω*x + 2im) / ω^3
    moment_x³_indefinite(x, ω) = exp(im*ω*x)*(-im*ω^3*x^3 + 3*ω^2*x^2 +6im*ω*x - 6) / ω^4

    moment_1(a, b, ω) = moment_1_indefinite(b, ω) - moment_1_indefinite(a, ω)
    moment_x(a, b, ω) = moment_x_indefinite(b, ω) - moment_x_indefinite(a, ω)
    moment_x²(a, b, ω) = moment_x²_indefinite(b, ω) - moment_x²_indefinite(a, ω)
    moment_x³(a, b, ω) = moment_x³_indefinite(b, ω) - moment_x³_indefinite(a, ω)

    @testset "1" begin
        @test moments[1] ≈ moment_1(a, b, ω)
        #@show moments[1] moment_1(a, b, ω)
    end
    @testset "x" begin
        @test moments[2] ≈ moment_x(a, b, ω)
        #@show moments[2] moment_x(a, b, ω)
    end
    @testset "x²" begin
        @test moments[3] ≈ moment_x²(a, b, ω)
        #@show moments[3] moment_x²(a, b, ω)
    end
    @testset "x³" begin
        @test moments[4] ≈ moment_x³(a, b, ω)
        #@show moments[4] moment_x³(a, b, ω)
    end
end

@testset "Filon weights agree with hard-coded weights" begin
    @testset "Interval [-1,1]" begin
        ω = 10.0
        a = -1 # For hardcoded, must have a=-1
        b = 1
        @testset "s=0" begin
            s=0
            weights_a, weights_b = filon_weights(ω, s, a, b)
            hard_weights_a, hard_weights_b = hardcoded_filon_weights(ω, s)
            @test weights_a[1] ≈ hard_weights_a[1]
            @test weights_b[1] ≈ hard_weights_b[1]
        end
        @testset "s=1" begin
            s=1
            weights_a, weights_b = filon_weights(ω, s, a, b)
            hard_weights_a, hard_weights_b = hardcoded_filon_weights(ω, s)
            @test weights_a[1] ≈ hard_weights_a[1]
            @test weights_a[2] ≈ hard_weights_a[2]
            @test weights_b[1] ≈ hard_weights_b[1]
            @test weights_b[2] ≈ hard_weights_b[2]
        end
    end
end
