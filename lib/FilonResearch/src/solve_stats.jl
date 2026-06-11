# Opt-in per-timestep diagnostics for the hard-coded and controlled Filon
# solvers.  The drivers thread the collector (or `nothing`) as a positional
# argument, so they specialize on its type and the `::Nothing` helper methods
# below compile away entirely — the default path reads no clock and records
# nothing.

"""
    FilonSolveStats()

Per-timestep diagnostics collector for [`filon_solve_hardcoded`](@ref) and
[`controlled_filon_solve`](@ref); pass it via their `stats` keyword.  Fields:

- `gmres_niters::Vector{Int}` — GMRES iteration count per step (dynamic
  variant only; empty for the static variant, which solves directly).
- `gmres_solved::Vector{Bool}` — GMRES convergence flag per step (dynamic only).
- `step_time_ns::Vector{UInt64}` — wall time per step in nanoseconds
  (`time_ns()` deltas).

The collector is emptied at the start of each solve, so one instance may be
reused across calls; after a solve it holds the most recent call's data only.

# Example

```julia
stats = FilonSolveStats()
ψ = filon_solve_hardcoded(co, ψ0, frequencies, Δt, nsteps, s; stats)
stats                      # pretty summary
stats.gmres_niters         # per-step iteration counts
```
"""
mutable struct FilonSolveStats
    gmres_niters::Vector{Int}
    gmres_solved::Vector{Bool}
    step_time_ns::Vector{UInt64}
end

FilonSolveStats() = FilonSolveStats(Int[], Bool[], UInt64[])

function Base.empty!(stats::FilonSolveStats)
    empty!(stats.gmres_niters)
    empty!(stats.gmres_solved)
    empty!(stats.step_time_ns)
    return stats
end

@inline _stats_init!(::Nothing, nsteps, dynamic::Bool) = nothing
function _stats_init!(stats::FilonSolveStats, nsteps, dynamic::Bool)
    empty!(stats)
    sizehint!(stats.step_time_ns, nsteps)
    if dynamic
        sizehint!(stats.gmres_niters, nsteps)
        sizehint!(stats.gmres_solved, nsteps)
    end
    return nothing
end

# Start-of-step marker; the `Nothing` path reads no clock.
@inline _stats_tick(::Nothing) = nothing
@inline _stats_tick(::FilonSolveStats) = time_ns()

@inline _stats_record_static!(::Nothing, t0) = nothing
@inline function _stats_record_static!(stats::FilonSolveStats, t0)
    push!(stats.step_time_ns, time_ns() - t0)
    return nothing
end

@inline _stats_record_dynamic!(::Nothing, t0, kws) = nothing
@inline function _stats_record_dynamic!(stats::FilonSolveStats, t0, kws)
    push!(stats.step_time_ns, time_ns() - t0)
    push!(stats.gmres_niters, Krylov.iteration_count(kws))
    push!(stats.gmres_solved, Krylov.issolved(kws))
    return nothing
end

Base.show(io::IO, stats::FilonSolveStats) =
    print(io, "FilonSolveStats(", length(stats.step_time_ns), " steps)")

function Base.show(io::IO, ::MIME"text/plain", stats::FilonSolveStats)
    n = length(stats.step_time_ns)
    print(io, "FilonSolveStats with ", n, " steps")
    total_s = sum(stats.step_time_ns; init = zero(UInt64)) / 1e9
    print(io, "\n  wall time: ", round(total_s; sigdigits = 4), " s total")
    n > 0 && print(io, ", ", round(total_s / n * 1e3; sigdigits = 4), " ms mean/step")
    if isempty(stats.gmres_niters)
        n > 0 && print(io, "\n  GMRES: not used (static variant)")
    else
        nit = stats.gmres_niters
        print(io, "\n  GMRES iterations: ", sum(nit), " total, ",
              round(sum(nit) / length(nit); sigdigits = 4), " mean, ",
              maximum(nit), " max")
        nfail = count(!, stats.gmres_solved)
        nfail > 0 && print(io, "\n  unconverged steps: ", nfail)
    end
end
