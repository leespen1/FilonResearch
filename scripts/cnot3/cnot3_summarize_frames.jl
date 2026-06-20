# Frame-aware variant of cnot3_summarize.jl: one row per (frame,
# initialCondition, method, s, nsteps).  cnot3_summarize.jl predates the frame
# dimension and omits `frame` from both the refinement-family key and the CSV,
# which silently mixes frames on multi-frame data; this script supersedes it
# there (data without a frame field is labeled "rwa").
#
#   julia --project scripts/cnot3/cnot3_summarize_frames.jl [prefix] [output.csv]
#
# Defaults: prefix = "cnot3Convergence", output = <this dir>/<prefix>_frames_summary.csv
# (a distinct name so the rwa-era <prefix>_summary.csv is not clobbered).
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
    joinpath(@__DIR__, "$(prefix)_frames_summary.csv")

outdir = commit_datadir(prefix)
df = collect_results(outdir)
isempty(df) && error("no result files found in $outdir")

"frame" in names(df) || (df.frame = fill("rwa", nrow(df)))
df.frame = string.(coalesce.(df.frame, "rwa"))

# Everything except nsteps and the saved outputs identifies a refinement family;
# pair each run with its nsteps/refinementFactor neighbour for the Richardson
# estimate (mirrors process_convergence_config).
family(r) = (r.frame, r.method, r.s, r.initialCondition, r.Tmax, r.nOscLevels,
             r.nGuardLevels, r.nsaves, r.refinementFactor)
history_of = Dict((family(r)..., r.nsteps) => r.history for r in eachrow(df))

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
    push!(rows, (; frame = r.frame, initialCondition = r.initialCondition,
                 method = r.method, s = r.s, order, nsteps = r.nsteps,
                 final_err, l2_err, t_elapsed = r.t_elapsed))
end

summary = sort!(DataFrame(rows), [:frame, :initialCondition, :method, :s, :nsteps])

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
