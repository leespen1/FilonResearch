using FilonResearch, Test
using StaticArrays

# Constant-coefficient 2×2 problem with matched static (SMatrix tuple) and
# dynamic (Vector) ControlledOperators, as in test_hardcoded_filon.jl.  The
# controls carry no carrier frequency, so the same operators also exercise
# controlled_filon_solve (zero-carrier coincides with regular Filon).
function stats_test_problem(ω)
    A = ComplexF64[im*ω 0; 1 im*ω]
    ctrls = (ConstantControl(1.0),)
    co_static = ControlledOperator(ctrls, (SMatrix{2,2}(A),))
    co_dynamic = ControlledOperator(ctrls, [A])
    return (; co_static, co_dynamic, frequencies = [ω, ω], ψ0 = ComplexF64[1.0, 1.0])
end

prob = stats_test_problem(10.0)
Δt_stats = 0.1
ns_stats = 10
s_stats = 1

@testset "dynamic variant collects iterations and times" begin
    for solver in (filon_solve_hardcoded, controlled_filon_solve)
        stats = FilonSolveStats()
        solver(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt_stats, ns_stats, s_stats;
               stats)
        @test length(stats.step_time_ns) == ns_stats
        @test length(stats.gmres_niters) == ns_stats
        @test length(stats.gmres_solved) == ns_stats
        @test all(>=(1), stats.gmres_niters)
        @test all(stats.gmres_solved)
    end
end

@testset "return value unchanged by stats collection" begin
    for solver in (filon_solve_hardcoded, controlled_filon_solve)
        for co in (prob.co_static, prob.co_dynamic)
            plain = solver(co, prob.ψ0, prob.frequencies, Δt_stats, ns_stats, s_stats)
            collected = solver(co, prob.ψ0, prob.frequencies, Δt_stats, ns_stats, s_stats;
                               stats = FilonSolveStats())
            @test collected ≈ plain
        end
    end
end

@testset "static variant collects times only" begin
    for solver in (filon_solve_hardcoded, controlled_filon_solve)
        stats = FilonSolveStats()
        solver(prob.co_static, prob.ψ0, prob.frequencies, Δt_stats, ns_stats, s_stats;
               stats)
        @test length(stats.step_time_ns) == ns_stats
        @test isempty(stats.gmres_niters)
        @test isempty(stats.gmres_solved)
    end
end

@testset "collector is emptied at each solve" begin
    stats = FilonSolveStats()
    filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt_stats,
                          ns_stats, s_stats; stats)
    filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt_stats,
                          5, s_stats; stats)
    @test length(stats.step_time_ns) == 5
    @test length(stats.gmres_niters) == 5
end

@testset "show" begin
    dyn_stats = FilonSolveStats()
    filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt_stats,
                          ns_stats, s_stats; stats = dyn_stats)
    static_stats = FilonSolveStats()
    filon_solve_hardcoded(prob.co_static, prob.ψ0, prob.frequencies, Δt_stats,
                          ns_stats, s_stats; stats = static_stats)

    @test occursin("FilonSolveStats", sprint(show, dyn_stats))
    dyn_text = sprint(show, MIME("text/plain"), dyn_stats)
    @test occursin("steps", dyn_text)
    @test occursin("GMRES iterations", dyn_text)
    static_text = sprint(show, MIME("text/plain"), static_stats)
    @test occursin("static variant", static_text)
    empty_text = sprint(show, MIME("text/plain"), FilonSolveStats())
    @test occursin("0 steps", empty_text)
end

@testset "GMRES non-convergence warns once per solve" begin
    # Unreachable tolerances: GMRES stops at itmax with issolved == false on
    # every step, but the warning must fire only once for the whole solve.
    for solver in (filon_solve_hardcoded, controlled_filon_solve)
        stats = FilonSolveStats()
        @test_logs (:warn, r"GMRES did not converge") begin
            solver(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt_stats, 5, s_stats;
                   gmres_atol = 1e-300, gmres_rtol = 1e-300, stats)
        end
        # For a 2×2 system GMRES can hit an exactly-zero residual once the
        # Krylov space is full, so some steps may still report solved.
        @test any(!, stats.gmres_solved)
    end
end

@testset "non-convergence warning is independent of stats" begin
    @test_logs (:warn, r"GMRES did not converge") begin
        filon_solve_hardcoded(prob.co_dynamic, prob.ψ0, prob.frequencies, Δt_stats,
                              5, s_stats; gmres_atol = 1e-300, gmres_rtol = 1e-300)
    end
end
