# Distil the CNOT3 convergence data into a compact, GitHub-friendly CSV: one row
# per run with the error against the Vern9 reference solution, the solve time, and
# the GMRES iteration statistics, dropping the bulky saved histories.  The CSV
# regenerates every convergence plot without keeping the .jld2 binaries under
# version control.
#
#   julia --project scripts/cnot3/cnot3_summarize.jl [prefix] [output.csv]
#
# Defaults: prefix = "cnot3Convergence", output = <this dir>/<prefix>_summary.csv.
#
# Errors are measured against the cached Vern9 reference for each
# (frame, initialCondition); precompute them first with cnot3_collect_reference.jl
# (this script only loads them, and errors if one is missing).

using DrWatson
@quickactivate "FilonExperiments"
using DataFrames
using Printf
using LinearAlgebra: norm
# Lightweight: reads jld2 only (no ODE solver).  The references must already be
# collected (see cnot3_collect_reference.jl).
include(srcdir("error_analysis.jl"))   # load_vern9_reference, reference_errors

const prefix = length(ARGS) >= 1 ? ARGS[1] : "cnot3Convergence"
const outcsv = length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "$(prefix)_summary.csv")

# black_list = String[] keeps the `gitcommit` column, which collect_results
# otherwise drops by default (it treats git provenance as bookkeeping).
outdir = datadir(prefix)
df = collect_results(outdir; black_list = String[])
isempty(df) && error("no result files found in $outdir")

# Cached Vern9 reference matching a run's frame / initial condition / problem size.
reference_for(r) = load_vern9_reference(; frame = r.frame, initialCondition = r.initialCondition,
                                        Nosc = r.nOscLevels, Nguard = r.nGuardLevels,
                                        Tmax = r.Tmax, nsaves = r.nsaves)

rows = NamedTuple[]
for r in eachrow(df)
    errs = reference_errors(r.history, reference_for(r), r.Tmax)
    push!(rows, (;
        initialCondition = r.initialCondition, frame = r.frame, method = r.method,
        s = r.s, order = 2 * (r.s + 1), nsteps = r.nsteps,
        final_err = errs.final_error, l2_err = errs.l2_error,
        t_elapsed = r.t_elapsed, nRuns = r.nRuns, saveFinalOnly = r.saveFinalOnly,
        gmres_mean = r.gmres_mean, gmres_median = r.gmres_median,
        gmres_max = r.gmres_max, gmres_std = r.gmres_std,
        gmresAtol = r.gmresAtol, gmresRtol = r.gmresRtol,
        gitcommit = r.gitcommit, gitdirty = r.gitdirty,
    ))
end

summary = sort!(DataFrame(rows),
                [:initialCondition, :frame, :method, :s, :gmresAtol, :gmresRtol, :nsteps])

# Minimal CSV writer (avoids adding a CSV.jl dependency).
csvcell(::Missing) = ""
csvcell(x) = string(x)
open(outcsv, "w") do io
    println(io, join(string.(names(summary)), ","))
    for r in eachrow(summary)
        println(io, join((csvcell(r[c]) for c in names(summary)), ","))
    end
end
@printf("Wrote %d rows to %s\n", nrow(summary), outcsv)
