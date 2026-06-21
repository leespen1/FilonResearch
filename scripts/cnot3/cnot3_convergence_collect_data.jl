"""
cnot3_convergence_collect_data.jl

Collect convergence data for four integrators on the optimized CNOT3 gate
problem from the High-Order Hermite Optimization (HOHO) paper:

  * `:hermite`            — QuantumGateDesign's `eval_forward` (the true Hermite
                            method, the baseline competitor).
  * `:filon`              — the new hard-coded Filon method
                            (`filon_solve_hardcoded`), drift-diagonal ansatz
                            frequencies.
  * `:controlled_filon`   — the new controlled Filon method
                            (`controlled_filon_solve`), which additionally factors
                            each control's carrier waves out of the quadrature.
  * `:controlled_hermite` — the ω = 0 counterpart of the efficient controlled
                            Filon method (`efficient_controlled_hermite_solve`):
                            same full A(t) as `:filon`, but applies each control
                            matrix only s+1 times per step.

Each `(method, s, nsteps)` solve is cached individually via DrWatson's
`produce_or_load`, so re-running only computes missing results.  Data collection
and plotting are separate: this script only writes per-run histories; see
`cnot3_plot.jl` for the figures.

# Running a subset (e.g. one SLURM batch)

With no arguments the full default sweep below runs (unchanged behaviour for a
plain `julia --project cnot3_convergence_collect_data.jl`).  Optional flags
narrow the sweep so a single job — local or on SLURM — can target a batch:

  * `--method hermite,filon`   one or more of
                               `hermite|filon|controlled_filon|controlled_hermite`
  * `--s 0,1`                  order parameter(s) s   (order = 2(s+1))
  * `--nsteps 128,256,512`     explicit step counts (overrides the 2^e defaults)
  * `--init basis,uniform`     initial condition(s): `basis` (default, full
                               essential basis / gate states), `uniform`, or `eN`
  * `--frame rwa,norwa,lab`    frame(s) the dynamics are posed in: `rwa`
                               (rotating frame + RWA, the original HOHO
                               setting), `norwa` (rotating frame keeping the
                               counter-rotating control terms), `lab`
                               (laboratory frame)

    julia --project cnot3_convergence_collect_data.jl --method hermite --nsteps 128,256,512
    sbatch cnot3_convergence_collect_data.sb --method hermite --init uniform

Because every run is cached by its config, subsets compose: running a few
batches separately fills the same data directory as one full sweep.
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
# "basis" (full essential computational basis — the gate states QGD propagates),
# "uniform" (uniform superposition), or "eN" (the N-th basis vector).
initialCondition = "basis"

all_methods = (:hermite, :filon, :controlled_filon, :controlled_hermite)
all_frames = ("rwa", "norwa", "lab")

s_values = (0, 1, 2)                       # order = 2(s+1) ∈ {2,4,6}
filon_step_exponents   = 4:16              # nsteps = 2^e for Filon / controlled-Filon
hermite_step_exponents = 4:22              # Hermite is cheap per step, push it further

# ------------------------------------------------------------
# Command-line selection (subset of the sweep above; see the docstring)
# ------------------------------------------------------------
# Parse `--flag value` pairs into an override for methods / s / nsteps; each is
# `nothing` when the flag is absent, in which case the default above is used.
function parse_selection(args)
    methods = s_sel = nsteps = inits = frames = nothing
    i = firstindex(args)
    while i <= lastindex(args)
        flag = args[i]
        i < lastindex(args) || throw(ArgumentError("flag '$flag' requires a value"))
        value = args[i + 1]
        if flag in ("--method", "--methods")
            methods = Symbol.(split(value, ','))
        elseif flag == "--s"
            s_sel = parse.(Int, split(value, ','))
        elseif flag == "--nsteps"
            nsteps = parse.(Int, split(value, ','))
        elseif flag == "--init"
            inits = String.(split(value, ','))
        elseif flag in ("--frame", "--frames")
            frames = String.(split(value, ','))
        else
            throw(ArgumentError(
                "unknown flag '$flag' (expected --method, --s, --nsteps, --init, or --frame)"))
        end
        i += 2
    end
    return (; methods, s_sel, nsteps, inits, frames)
end

selection = parse_selection(ARGS)

selected_methods  = something(selection.methods, collect(all_methods))
selected_s_values = something(selection.s_sel, collect(s_values))
selected_inits    = something(selection.inits, [initialCondition])
selected_frames   = something(selection.frames, collect(all_frames))

for m in selected_methods
    m in all_methods ||
        throw(ArgumentError("unknown method ':$m'; choose from $all_methods"))
end
for ic in selected_inits
    ic in ("basis", "uniform") || occursin(r"^e\d+$", ic) ||
        throw(ArgumentError("unknown initial condition '$ic'; " *
            "choose from \"basis\", \"uniform\", or \"eN\""))
end
for fr in selected_frames
    fr in all_frames ||
        throw(ArgumentError("unknown frame '$fr'; choose from $all_frames"))
end

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
for initialCondition in selected_inits, frame in selected_frames, method in selected_methods
    # Explicit --nsteps overrides the per-method 2^e defaults.
    step_counts = if selection.nsteps !== nothing
        selection.nsteps
    else
        exponents = method === :hermite ? hermite_step_exponents : filon_step_exponents
        2 .^ exponents
    end
    for s in selected_s_values, nsteps in step_counts
        mod(nsteps, nsaves) == 0 ||
            throw(ArgumentError("nsteps=$nsteps must be divisible by nsaves=$nsaves"))
        push!(configs, (;
            method, frame, s, Tmax, initialCondition, nOscLevels, nGuardLevels,
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
