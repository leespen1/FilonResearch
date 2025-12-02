using Test

@testset "Hermite cardinal polynomials" begin
    include("./test_hermite_cardinal_polynomials.jl")
end
@testset "Filon moments and weights" begin
    include("./test_filon_weights.jl")
end
@testset "Explicit Filon integration" begin
    include("./test_explcit_filon_integration.jl")
end

