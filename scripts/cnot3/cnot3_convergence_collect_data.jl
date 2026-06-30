"""
cnot3_convergence_collect_data.jl

Collect convergence data for the CNOT3 gate problem from the High-Order Hermite
Optimization (HOHO) paper.  Six FilonResearch integrators are available, in three
naive/efficient pairs — each pair is the same method to ~1e-14, the efficient
variant reorganizing the operator apply (each control matrix applied s+1 times):

  * `:Hermite` / `:NaiveHermite`               — ω = 0 Hermite
    (`efficient_controlled_hermite_solve` / `hermite_solve_hardcoded`).
  * `:Filon` / `:NaiveFilon`                   — Filon, drift-diagonal ansatz
    (`efficient_filon_solve` / `filon_solve_hardcoded`).
  * `:ControlledFilon` / `:NaiveControlledFilon` — controlled Filon, carriers
    factored out (`efficient_controlled_filon_solve` / `controlled_filon_solve`).

A seventh method, `:HermiteQGD`, propagates with QuantumGateDesign's own
`eval_forward` Hermite solver at order 2(s+1); it is a baseline the hand-rolled
`:Hermite` is compared against.

The default sweep collects the three efficient variants plus `:HermiteQGD`; the
`Naive*` ones are available via `--method` for naive-vs-efficient timing
comparisons.

Each run is cached individually via DrWatson's `produce_or_load`, so re-running
only computes missing results.  Each result file records the git commit it was
produced on (and whether the working tree was dirty) for provenance, but the
data directory is *not* namespaced by commit — runs are reused across commits.

Data collection and plotting are separate: this script only writes per-run
histories and timing/GMRES diagnostics.  Errors are computed downstream against a
Vern9 reference solution (see `cnot3_summarize.jl` / `cnot3_plot.jl`); nothing
about a run's accuracy is stored here.

# Running a subset (e.g. one SLURM batch)

With no arguments the full default sweep below runs.  Optional flags narrow the
sweep so a single job — local or on SLURM — can target a batch:

  * `--method Filon,Hermite`   one or more of `NaiveHermite|Hermite|NaiveFilon|
                               Filon|NaiveControlledFilon|ControlledFilon|
                               HermiteQGD` (the last propagates with
                               QuantumGateDesign's own `eval_forward` Hermite
                               solver, as a baseline for `Hermite`)
  * `--s 0,1`                  order parameter(s) s   (order = 2(s+1))
  * `--nsteps 128,256,512`     explicit step counts (overrides the 2^e defaults)
  * `--init basis,uniform`     initial condition(s): `basis` (default, full
                               essential basis / gate states) or `uniform`
                               (uniform superposition)
  * `--frame rwa,norwa,lab`    frame(s) the dynamics are posed in: `rwa`
                               (rotating frame + RWA, the original HOHO
                               setting), `norwa` (rotating frame keeping the
                               counter-rotating control terms), `lab`
                               (laboratory frame)
  * `--gmres-atol 1e-13,1e-10` GMRES absolute tolerance(s) for the iterative
                               (Filon-family) solvers
  * `--gmres-rtol 1e-13,1e-10` GMRES relative tolerance(s)
  * `--nruns 5`                repeat each solve this many times and average the
                               timing (default 1); the result history is
                               unchanged, only the recorded t_elapsed is averaged
  * `--save-final-only true`   save only the final state instead of the full
                               N × (nsaves+1) history (the default, false, which
                               the l2-integral error needs)

    julia --project cnot3_convergence_collect_data.jl --method Hermite --nsteps 128,256,512
    sbatch cnot3_convergence_collect_data.sb --method Filon --gmres-rtol 1e-13,1e-8

The GMRES tolerances are part of each run's identity (they appear in the
savename), so a tolerance sweep produces distinct cached files.

Because every run is cached by its config, subsets compose: running a few batches
separately fills the same data directory as one full sweep.
"""

# ============================================================
# Experiment parameters
# ============================================================

# Subsystem dimensions and pulse duration (full HOHO settings).
nOscLevels = 10
nGuardLevels = 2
Tmax = 550.0

# Number of saved time points (after the initial condition).  Every nsteps is a
# power of two ≥ this, so nsaves divides nsteps and the saved grid is shared.
nsaves = 16
# "basis" (full essential computational basis — the gate states QGD propagates)
# or "uniform" (uniform superposition).
initialCondition = "basis"

# The six solvers (three naive/efficient pairs); the default sweep uses the three
# efficient ones, with the naive variants opt-in via --method.
all_methods     = (:NaiveHermite, :Hermite, :NaiveFilon, :Filon,
                   :NaiveControlledFilon, :ControlledFilon, :HermiteQGD)
default_methods = (:Hermite, :Filon, :ControlledFilon, :HermiteQGD)
all_frames = ("rwa", "norwa", "lab")

s_values = (0, 1, 2)                       # order = 2(s+1) ∈ {2,4,6}
step_exponents = 4:16                      # nsteps = 2^e; deeper runs via --nsteps

# GMRES tolerances for the iterative solvers (a single tight value by default).
gmres_atols = (1e-13,)
gmres_rtols = (1e-13,)

# Number of timed repetitions per solve; t_elapsed is averaged over them.  The
# result history is identical across repetitions, so this only sharpens timing.
nRuns = 1

# Whether to save only the final state (true) or the full N × (nsaves+1) save
# grid (false, the default — the l2-integral error needs the full grid).
saveFinalOnly = false

# ------------------------------------------------------------
# Command-line selection (subset of the sweep above; see the docstring)
# ------------------------------------------------------------
# Parse `--flag value` pairs into overrides; each field is `nothing` when the
# corresponding flag is absent, in which case the default above is used.
function parse_selection(args)
    methods = s_sel = nsteps = inits = frames = atols = rtols = nruns = sfo = nothing
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
        elseif flag == "--gmres-atol"
            atols = parse.(Float64, split(value, ','))
        elseif flag == "--gmres-rtol"
            rtols = parse.(Float64, split(value, ','))
        elseif flag == "--nruns"
            nruns = parse(Int, value)
        elseif flag == "--save-final-only"
            sfo = parse(Bool, value)
        else
            throw(ArgumentError("unknown flag '$flag' (expected --method, --s, " *
                "--nsteps, --init, --frame, --gmres-atol, --gmres-rtol, --nruns, " *
                "or --save-final-only)"))
        end
        i += 2
    end
    return (; methods, s_sel, nsteps, inits, frames, atols, rtols, nruns, sfo)
end

selection = parse_selection(ARGS)

selected_methods  = something(selection.methods, collect(default_methods))
selected_s_values = something(selection.s_sel, collect(s_values))
selected_inits    = something(selection.inits, [initialCondition])
selected_frames   = something(selection.frames, collect(all_frames))
selected_atols    = something(selection.atols, collect(gmres_atols))
selected_rtols    = something(selection.rtols, collect(gmres_rtols))
selected_nRuns    = something(selection.nruns, nRuns)
selected_sfo      = something(selection.sfo, saveFinalOnly)

selected_nRuns >= 1 || throw(ArgumentError("nRuns must be ≥ 1; got $selected_nRuns"))
for m in selected_methods
    m in all_methods ||
        throw(ArgumentError("unknown method ':$m'; choose from $all_methods"))
end
for ic in selected_inits
    ic in ("basis", "uniform") ||
        throw(ArgumentError("unknown initial condition '$ic'; choose \"basis\" or \"uniform\""))
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
@everywhere const outdir = datadir(prefix)

@everywhere println("[", myid(), "] Finished setting up helper functions.")
@everywhere flush(stdout)

# ============================================================
# Collect configs and run
# ============================================================

configs = NamedTuple[]
for initialCondition in selected_inits, frame in selected_frames, method in selected_methods,
        gmresAtol in selected_atols, gmresRtol in selected_rtols
    # Explicit --nsteps overrides the 2^e defaults.
    step_counts = selection.nsteps !== nothing ? selection.nsteps : 2 .^ step_exponents
    for s in selected_s_values, nsteps in step_counts
        mod(nsteps, nsaves) == 0 ||
            throw(ArgumentError("nsteps=$nsteps must be divisible by nsaves=$nsaves"))
        push!(configs, (;
            method, frame, s, Tmax, initialCondition, nOscLevels, nGuardLevels,
            nsaves, saveFinalOnly = selected_sfo, nRuns = selected_nRuns,
            gmresAtol, gmresRtol, nsteps,
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
