"""
cnot3_convergence_collect_data.jl

Collect convergence data for three integrators on the optimized CNOT3 gate
problem from the High-Order Hermite Optimization (HOHO) paper:

  * `:hermite`          — QuantumGateDesign's `eval_forward` (the true Hermite
                          method, the baseline competitor).
  * `:filon`            — the new hard-coded Filon method
                          (`filon_solve_hardcoded`), drift-diagonal ansatz
                          frequencies.
  * `:controlled_filon` — the new controlled Filon method
                          (`controlled_filon_solve`), which additionally factors
                          each control's carrier waves out of the quadrature.

Each `(method, s, nsteps)` solve is cached individually via DrWatson's
`produce_or_load`, so re-running only computes missing results.  Data collection
and plotting are separate: this script only writes per-run histories; see
`cnot3_plot.jl` for the figures.
"""

# ============================================================
# Experiment parameters
# ============================================================

# Subsystem dimensions and pulse duration (full HOHO settings).
nOscLevels = 10
nGuardLevels = 2
Tmax = 550.0

refinementFactor = 2

# Number of saved time points (after the initial condition).  Every nsteps is a
# power of two ≥ this, so nsaves divides nsteps and the saved grid is shared.
nsaves = 16
initialCondition = "uniform" # "uniform", or "eN" for the N-th basis vector

s_values = (0, 1, 2)                       # order = 2(s+1) ∈ {2,4,6}
filon_step_exponents   = 4:16              # nsteps = 2^e for Filon / controlled-Filon
hermite_step_exponents = 4:22              # Hermite is cheap per step, push it further

# ============================================================
# Set up distributed environment
# ============================================================
using Distributed, SlurmClusterManager

if haskey(ENV, "SLURM_JOBID") || haskey(ENV, "SLURM_JOB_ID")
    addprocs(SlurmManager())
end

@everywhere using DrWatson
@everywhere @quickactivate "FilonExperiments"
@everywhere using FilonResearch
@everywhere using QuantumGateDesign
@everywhere using Printf
@everywhere using LinearAlgebra: norm
@everywhere include(srcdir("error_analysis.jl"))   # process_convergence_config
@everywhere include(srcdir("cnot3_run.jl"))        # make_initial_condition, run_simulation

@everywhere const prefix = "cnot3Convergence"
@everywhere const outdir = commit_datadir(prefix)

@everywhere println("[", myid(), "] Finished setting up helper functions.")
@everywhere flush(stdout)

# ============================================================
# Collect configs and run
# ============================================================

configs = NamedTuple[]
for method in (:hermite, :filon, :controlled_filon)
    exponents = method === :hermite ? hermite_step_exponents : filon_step_exponents
    for s in s_values, e in exponents
        nsteps = 2^e
        push!(configs, (;
            method, s, Tmax, initialCondition, nOscLevels, nGuardLevels,
            nsaves, refinementFactor, nsteps,
        ))
    end
end

run_successes = pmap(configs) do config
    try
        process_convergence_config(run_simulation, config, prefix, outdir)
    catch ex
        @warn "Simulation failed" config exception=(ex, catch_backtrace())
        false
    end
end

if !all(run_successes)
    println("\n")
    @warn "Not all runs were successful!"
    for i in findall(!, run_successes)
        println("\t", configs[i])
    end
    flush(stdout)
end

# Re-run to print a summary table (all simulations are now cached).
println("\n"^3, "-"^80, "\n"^3, "Finished running simulations. Printing summary table.\n")
map(configs) do config
    try
        process_convergence_config(run_simulation, config, prefix, outdir)
    catch ex
        @warn "Simulation failed" config exception=(ex, catch_backtrace())
        false
    end
end
flush(stdout)
