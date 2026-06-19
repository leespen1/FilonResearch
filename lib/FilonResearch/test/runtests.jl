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

# NOTE: test_hardcoded_lhs_rhs.jl targeted the removed src/hardcoded_lhs_rhs.jl
# (replaced by the ControlledOperator-based hard-coded Filon method below).
@testset "Hard-coded Filon (ControlledOperator)" begin
    include("./test_hardcoded_filon.jl")
end

@testset "Hard-coded Hermite (ControlledOperator)" begin
    include("./test_hardcoded_hermite.jl")
end

@testset "Controlled Filon (Appendix B)" begin
    include("./test_controlled_filon.jl")
end

@testset "Efficient controlled Filon (Appendix B)" begin
    include("./test_efficient_controlled_filon.jl")
end

@testset "Efficient controlled Hermite (ω = 0)" begin
    include("./test_efficient_controlled_hermite.jl")
end

@testset "Solve diagnostics (FilonSolveStats)" begin
    include("./test_solve_stats.jl")
end

