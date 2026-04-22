"""
cnot3_collect_data.jl

Collect convergence data for Filon and Hermite methods on CNOT3 gate problem.
Each (method, s, nsteps) solve is cached individually via DrWatson's
produce_or_load, so re-running only computes missing results.
"""


# ============================================================
# Experiment parameters - Frequently Changed
# ============================================================

# Subsystem dimensions
nOscLevels = 10 # default 10
nGuardLevels = 2 # default 2
Tmax = 100.0 # default 550.0

refinementFactor = 2

# For each s-value of the Filon method, run test for various numbers of timesteps
# (if timesteps are floats, will be rounded *up*)
filon_s_to_nsteps = (
    0 => refinementFactor .^ (2:16),
    1 => refinementFactor .^ (2:16),
    2 => refinementFactor .^ (2:16),
)

hermite_s_to_nsteps = (
    0 => refinementFactor .^ (2:22),
    1 => refinementFactor .^ (2:22),
    2 => refinementFactor .^ (2:22),
)

nsaves = 16 # Number of time points (after initial condition) to save to jld2 files
initialCondition = "uniform" # uniform, or eN, where N is an integer

# ============================================================
# Set up distributed environment
# ============================================================
using Distributed, SlurmClusterManager

if haskey(ENV, "SLURM_JOBID") || haskey(ENV, "SLURM_JOB_ID")
    addprocs(SlurmManager())
end


# ============================================================
# Set up functions for running simulations
# ============================================================

@everywhere using DrWatson
@everywhere @quickactivate "FilonExperiments"
@everywhere using FilonResearch
@everywhere using QuantumGateDesign
@everywhere using Printf
@everywhere using LinearAlgebra: norm, diag
@everywhere include(srcdir("error_analysis.jl"))
@everywhere include(srcdir("cnot3_hoho_helpers.jl"))
@everywhere include(srcdir("QuantumGateDesign_interface.jl"))

@everywhere const prefix = "cnot3Convergence"
@everywhere const outdir = datadir(prefix)

# Given a 'spec' and the system size, construct the initial condition as a vector
@everywhere function make_initial_condition(spec, n::Integer)
    if spec == "uniform" # Uniform superposition
        u0 = ones(ComplexF64, n)
        return u0 ./ norm(u0)
    elseif spec isa AbstractString # Basis vector (string parsed version)
        m = match(r"^e(\d+)$", spec)
        if m !== nothing
            k = parse(Int, m.captures[1])
            1 <= k <= n || throw(ArgumentError("basis index must be between 1 and $n"))
            u0 = zeros(ComplexF64, n)
            u0[k] = 1.0
            return u0
        end
    end

    throw(ArgumentError("unknown initial condition spec: $spec"))
end

#The main simulation
#Solve the ODE with a given config.
@everywhere function run_simulation(config)
    qgd_prob = cnot3_hoho_qgd_prob(
        N_osc_levels = config.nOscLevels,
        N_guard_levels = config.nGuardLevels,
        Tmax = config.Tmax
    ) 
    # Parse config
    initial_condition = make_initial_condition(config.initialCondition, qgd_prob.N_tot_levels)

    # Infer from config
    order = 2*(config.s+1)
    t_saves = collect(range(0, config.Tmax, length=1+config.nsaves))

    controls, pcof = cnot3_hoho_controls_and_pcof()

    if config.method == :hermite
        saveEveryNsteps, remainder = divrem(config.nsteps, config.nsaves)
        remainder == 0 || throw(ArgumentError("Number of saves ($(config.nsaves)) and number of timesteps ($(config.nsteps)) are incompatible (remainder = $remainder)."))

        prob = QuantumGateDesign.VectorSchrodingerProb(qgd_prob, 1)
        prob.u0 .= real(initial_condition)
        prob.v0 .= imag(initial_condition)

        # Run once to get compilation out of the way
        prob.nsteps = 1
        t_elapsed = @elapsed history = eval_forward(
            prob, controls, pcof, order=order,
        )
        # Run again, for real this time
        prob.nsteps = config.nsteps
        t_elapsed = @elapsed history = eval_forward(
            prob, controls, pcof, order=order, saveEveryNsteps=saveEveryNsteps,
        )
    elseif config.method == :filon
        A_deriv_funcs = QGD_prob_to_filon_hamiltonian(qgd_prob, controls, pcof, config.s)
        filon_freqs = -1.0 .* Array(diag(qgd_prob.system_sym))

        # Run once to get compilation out of the way
        dummy_nsteps = 1
        t_elapsed = @elapsed history = filon_solve(
            A_deriv_funcs, initial_condition, filon_freqs, config.Tmax, dummy_nsteps, config.s,
        )
        # Run again, for real this time
        t_elapsed = @elapsed history = filon_solve(
            A_deriv_funcs, initial_condition, filon_freqs, config.Tmax, config.nsteps, config.s,
        )
    else
        throw(ArgumentError("Invalid method '$(config.method)'. Method must be either 'hermite' or 'filon'."))
    end

    # Downsample history and store
    history = downsample_history(history, config.nsaves)

    return @strdict history t_elapsed t_saves
end

@everywhere println("[", myid(), "]", " Finished setting up helper functions.")
@everywhere flush(stdout)

# ============================================================
# Run the simulations
# ============================================================

# Collect simulation configs
configs = []

for (s, nsteps_vec) in filon_s_to_nsteps
    for nsteps_maybe_float in nsteps_vec
        nsteps = round(Int, nsteps_maybe_float) 
        # Create config as a NamedTuple [`(; x, y, ...)` infers variable names from arguments]
        config = (;
           method=:filon,
           s,
           Tmax,
           initialCondition,
           nOscLevels,
           nGuardLevels,
           nsaves,
           refinementFactor,
           nsteps,
        )
        push!(configs, config)
    end
end

for (s, nsteps_vec) in hermite_s_to_nsteps
    for nsteps_maybe_float in nsteps_vec
        nsteps = round(Int, nsteps_maybe_float) 
        # Create config as a NamedTuple [`(; x, y, ...)` infers variable names from arguments]
        config = (;
           method=:hermite,
           s,
           Tmax,
           initialCondition,
           nOscLevels,
           nGuardLevels,
           nsaves,
           refinementFactor,
           nsteps,
        )
        push!(configs, config)
    end
end


run_successes = pmap(configs) do config
    try
        process_convergence_config(config)
    catch ex
        @warn "Simulation failed" config exception=(ex, catch_backtrace())
        false
    end
end

if !all(run_successes)
    println("\n")
    @warn "Not all runs were successful!"
    idx = findall(!, run_successes)
    println("Unsuccessful runs:")
    for i in idx
        println("\t", configs[i])
    end
    flush(stdout)
end


# Run a second time. Since all the simulations have already been run, this just prints a summary of the results.
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
