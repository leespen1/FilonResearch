# Distil the CNOT3 convergence data into a compact, GitHub-friendly CSV: one row
# per (initialCondition, method, s, nsteps) with the Richardson error estimates
# and the solve time, dropping the bulky saved histories.  The CSV regenerates
# every convergence plot without keeping the .jld2 binaries under version control.
#
#   julia --project scripts/cnot3/cnot3_summarize.jl [prefix] [output.csv]
#
# Defaults: prefix = "cnot3Convergence", output = <this dir>/<prefix>_summary.csv.
# Reads the commit-namespaced data dir (commit_datadir), so it summarizes exactly
# the runs produced by the current checkout.

using DrWatson
@quickactivate "FilonExperiments"
using DataFrames
using Printf
using LinearAlgebra: norm
include(srcdir("error_analysis.jl"))   # richardson_error, richardson_l2_integral_error

const prefix = length(ARGS) >= 1 ? ARGS[1] : "cnot3Convergence"
const outcsv = length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "$(prefix)_summary.csv")

outdir = commit_datadir(prefix)
df = collect_results(outdir)
isempty(df) && error("no result files found in $outdir")

# Everything except nsteps and the saved outputs identifies a refinement family;
# pair each run with its nsteps/refinementFactor neighbour for the Richardson
# estimate (mirrors process_convergence_config).  `frame` is part of the family:
# the three frames pose different dynamics, so a run is only comparable to its
# same-frame neighbour.
family(r) = (r.method, r.frame, r.s, r.initialCondition, r.Tmax, r.nOscLevels,
             r.nGuardLevels, r.nsaves, r.refinementFactor)
history_of = Dict((family(r)..., r.nsteps) => r.history for r in eachrow(df))

# Older data dirs predate the GMRES iteration tracking; tolerate its absence.
has_gmres = ("avg_gmres" in names(df)) && ("max_gmres" in names(df))
gmres(r, col) = has_gmres ? r[col] : missing

rows = NamedTuple[]
for r in eachrow(df)
    order = 2 * (r.s + 1)
    coarse_nsteps = div(r.nsteps, r.refinementFactor)
    prev = get(history_of, (family(r)..., coarse_nsteps), nothing)
    if prev === nothing
        final_err = l2_err = missing
    else
        final_err = richardson_error(
            r.history[:, end], prev[:, end], r.nsteps, coarse_nsteps, order)
        l2_err = richardson_l2_integral_error(
            r.history, prev, r.nsteps, coarse_nsteps, r.Tmax, order)
    end
    push!(rows, (; initialCondition = r.initialCondition, frame = r.frame,
                 method = r.method, s = r.s, order, nsteps = r.nsteps,
                 final_err, l2_err, t_elapsed = r.t_elapsed,
                 avg_gmres = gmres(r, :avg_gmres), max_gmres = gmres(r, :max_gmres)))
end

summary = sort!(DataFrame(rows), [:initialCondition, :frame, :method, :s, :nsteps])

# Minimal CSV writer (avoids adding a CSV.jl dependency).
csvcell(::Missing) = ""
csvcell(x) = string(x)
open(outcsv, "w") do io
    println(io, join(string.(names(summary)), ","))
    for r in eachrow(summary)
        println(io, join((csvcell(r[c]) for c in names(summary)), ","))
    end
end
@printf("Wrote %d rows (%d with a Richardson estimate) to %s\n",
        nrow(summary), count(!ismissing, summary.final_err), outcsv)
