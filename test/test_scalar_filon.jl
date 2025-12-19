using FilonResearch, Test
import Random: MersenneTwister

@testset "Scalar Filon" begin
@testset "Dahlquist, exact solves for single timestep, λ=iω" begin
    @testset "Rescale=$rescale" for rescale in (false, true)
        t_tups = [(-1, 1), (0, 1), (-0.9, 0.7)]
        @testset "t ∈ [$(t_tup[1]), $(t_tup[2])]" for t_tup in t_tups 
            tₙ, tₙ₊₁ = t_tup
            λ_ω_uₙ_tups = [
                (im, 1, 1),
                (im, 1, 2+3im),
                (5im, 5, 1),
                (5im, 5, 2+3im),
            ]
            @testset "λ=$(tup[1]), ω=$(tup[2]), u(-1) = $(tup[3])" for tup in λ_ω_uₙ_tups
                λ, ω, uₙ = tup
                true_sol = uₙ * exp(λ*(tₙ₊₁ - tₙ))
                @testset "s=$s" for s in 0:3
                    uₙ₊₁ = filon_timestep(λ, ω, uₙ, s, tₙ, tₙ₊₁; rescale=rescale)
                    @test uₙ₊₁ ≈ true_sol rtol=1e-13
                end
            end
        end
    end
end

# These tests involve use more difficult time intervals, e.g. large intervals,
# short intervals, short intervals at large values of t.
@testset "Dahlquist, single timestep, λ=iω, difficult t" begin
    @testset "Rescale=$rescale" for rescale in (false, true)
        t_tups = [(-1.5, 2), (1e-5, 1.1e-5), (1e4, 1e4+1e-5), (-100, 200)]
        @testset "t ∈ [$(t_tup[1]), $(t_tup[2])]" for t_tup in t_tups 
            tₙ, tₙ₊₁ = t_tup
            λ_ω_uₙ_tups = [
                (im, 1, 1),
                (im, 1, 2+3im),
                (5im, 5, 1),
                (5im, 5, 2+3im),
            ]
            @testset "λ=$(tup[1]), ω=$(tup[2]), u(-1) = $(tup[3])" for tup in λ_ω_uₙ_tups
                λ, ω, uₙ = tup
                true_sol = uₙ * exp(λ*(tₙ₊₁ - tₙ))
                @testset "s=$s" for s in 0:3
                    uₙ₊₁ = filon_timestep(λ, ω, uₙ, s, tₙ, tₙ₊₁; rescale=rescale)
                    @test uₙ₊₁ ≈ true_sol rtol=1e-10 # higher tolerance for more difficult t
                end
            end
        end
    end
end
end # @testset "Scalar Filon"
