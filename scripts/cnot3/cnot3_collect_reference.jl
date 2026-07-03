"""
cnot3_collect_reference.jl

Precompute and cache the Vern9 reference solutions used to measure CNOT3
convergence error, so they exist as jld2 files (in `datadir("cnot3_vern9ref")`)
ahead of any summarize / plot / paper run — those scripts then just load them
instead of computing the (slow, lab-frame) reference on the fly.

Each `(frame, initialCondition)` reference is cached via DrWatson's
`produce_or_load`, so re-running only computes the missing ones.

After collecting, prints the CNOT gate fidelity/infidelity of each selected
frame's "basis" reference (a cheap post-processing of the cached final state).

Optional flags narrow the set (default: all frames × both initial conditions):
  * `--frame rwa,norwa,lab`
  * `--init  basis,uniform`

    julia --project scripts/cnot3/cnot3_collect_reference.jl
    sbatch scripts/cnot3/cnot3_collect_reference.sb --frame lab
"""

# Problem size and reference tolerance — must match the collected sweep.
const NOSC   = 10
const NGUARD = 2
const TMAX   = 550.0
const NSAVES = 16
const ABSTOL = 1e-15
const RELTOL = 1e-15

all_frames = ("rwa", "norwa", "lab")
all_inits  = ("basis", "uniform")

# ------------------------------------------------------------
# Command-line selection (subset of the full set above)
# ------------------------------------------------------------
function parse_selection(args)
    frames = inits = nothing
    i = firstindex(args)
    while i <= lastindex(args)
        flag = args[i]
        i < lastindex(args) || throw(ArgumentError("flag '$flag' requires a value"))
        value = args[i + 1]
        if flag in ("--frame", "--frames")
            frames = String.(split(value, ','))
        elseif flag == "--init"
            inits = String.(split(value, ','))
        else
            throw(ArgumentError("unknown flag '$flag' (expected --frame or --init)"))
        end
        i += 2
    end
    return (; frames, inits)
end

selection = parse_selection(ARGS)
selected_frames = something(selection.frames, collect(all_frames))
selected_inits  = something(selection.inits, collect(all_inits))
for fr in selected_frames
    fr in all_frames || throw(ArgumentError("unknown frame '$fr'; choose from $all_frames"))
end
for ic in selected_inits
    ic in all_inits || throw(ArgumentError("unknown initial condition '$ic'; choose from $all_inits"))
end

# ============================================================
# Distributed environment
# ============================================================
using Distributed, SlurmClusterManager

if haskey(ENV, "SLURM_JOBID") || haskey(ENV, "SLURM_JOB_ID")
    addprocs(SlurmManager())
end

@everywhere using DrWatson
@everywhere @quickactivate "FilonExperiments"
@everywhere using FilonResearch
@everywhere using QuantumGateDesign
@everywhere using LinearAlgebra: norm
using Printf
@everywhere include(srcdir("cnot3_run.jl"))        # problem builders, make_initial_condition
@everywhere include(srcdir("cnot3_reference.jl"))  # vern9_reference

# ============================================================
# Compute (or load) each reference
# ============================================================
configs = [(; frame, init) for frame in selected_frames for init in selected_inits]
println("Collecting $(length(configs)) Vern9 references into ", datadir("cnot3_vern9ref"))
flush(stdout)

results = pmap(configs) do c
    try
        data = vern9_reference(; frame = c.frame, initialCondition = c.init,
            Nosc = NOSC, Nguard = NGUARD, Tmax = TMAX, nsaves = NSAVES,
            abstol = ABSTOL, reltol = RELTOL)
        println("[", myid(), "] done frame=", rpad(c.frame, 6), " init=", rpad(c.init, 8),
                " ‖uref‖=", round(norm(data["uref"]); sigdigits = 6))
        flush(stdout)
        true
    catch ex
        @warn "Reference failed" config=c exception=(ex, catch_backtrace())
        false
    end
end

if !all(results)
    @warn "Not all references succeeded!"
    for i in findall(!, results)
        println("\t", configs[i])
    end
end

# ============================================================
# CNOT gate fidelity of each frame's "basis" reference
# ============================================================
if "basis" in selected_inits
    println("\nCNOT gate fidelity vs Vern9 basis reference (F = |tr(target† U)|²/N_ess²):")
    for frame in selected_frames
        data = vern9_reference(; frame, initialCondition = "basis",
            Nosc = NOSC, Nguard = NGUARD, Tmax = TMAX, nsaves = NSAVES,
            abstol = ABSTOL, reltol = RELTOL)          # cache hit: loads the jld2
        F = cnot3_gate_fidelity(data["uref"]; frame,
            Nosc = NOSC, Nguard = NGUARD, Tmax = TMAX)
        @printf "  frame=%-6s  fidelity=%.12f  infidelity=%.6e\n" frame F (1 - F)
    end
end
flush(stdout)
println("Done.")
