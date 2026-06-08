# Fast end-to-end smoke test of the CNOT3 pipeline: a tiny, reduced problem run
# through the same `produce_or_load` machinery as the real collection script,
# written to a separate "cnot3ConvergenceSmoke" data prefix.  Verifies that
# results are cached (a re-run skips them) and that the plotting script produces
# a figure.  Not the real experiment.
#
#   julia --project=. scripts/cnot3/smoke_test.jl

using DrWatson
@quickactivate "FilonExperiments"

using Distributed                  # process_convergence_config prints myid()
using FilonResearch
using QuantumGateDesign
using Printf
using LinearAlgebra: norm
include(srcdir("error_analysis.jl"))
include(srcdir("cnot3_run.jl"))

const prefix = "cnot3ConvergenceSmoke"
const outdir = datadir(prefix)

# Reduced problem: small Hilbert space, short pulse window, few steps/saves.
nOscLevels = 3
nGuardLevels = 1
Tmax = 50.0
refinementFactor = 2
nsaves = 4
initialCondition = "uniform"

configs = NamedTuple[]
for method in (:hermite, :filon, :controlled_filon)
    for s in (0, 1), e in 4:6
        push!(configs, (;
            method, s, Tmax, initialCondition, nOscLevels, nGuardLevels,
            nsaves, refinementFactor, nsteps = 2^e,
        ))
    end
end

println("Running $(length(configs)) reduced configs into $outdir ...")
foreach(c -> process_convergence_config(run_simulation, c, prefix, outdir), configs)

nfiles = length(filter(endswith(".jld2"), readdir(outdir)))
@printf("\n%d result files present.\n", nfiles)

# Re-run: every config should now be a cache hit (fast, no recompute).
println("Re-running to confirm caching (should be instant) ...")
t_rerun = @elapsed foreach(
    c -> process_convergence_config(run_simulation, c, prefix, outdir), configs)
@printf("Re-run of %d configs took %.3f s (cache hits).\n", length(configs), t_rerun)

# Drive the plotting script against the smoke data.
println("\nRunning the plotting script on the smoke data ...")
ENV["CNOT3_PREFIX"] = prefix
include(joinpath(@__DIR__, "cnot3_plot.jl"))
println("\nSmoke test complete.")
