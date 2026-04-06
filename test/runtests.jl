using Test

@testset "Hermite cardinal polynomials" begin
    include("./test_hermite_cardinal_polynomials.jl")
end
@testset "Filon moments and weights" begin
    include("./test_filon_weights.jl")
end
@testset "Explicit Filon integration" begin
    include("./test_explicit_filon_integration.jl")
end
@testset "Manufactured polynomial solution" begin
    include("./test_manufactured_polynomial_solution.jl")
end

@testset "Manufactured polynomial solution, multifrequency" begin
    include("./test_manufactured_polynomial_solution_multifrequency.jl")
end

@testset "Hard-coded LHS/RHS" begin
    include("./test_hardcoded_lhs_rhs.jl")
end

