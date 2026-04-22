"""
Numerical experiments on the Rabi oscillation problem.

Collect convergence data for Filon and Hermite (implemented as Filon) methods
for a Rabi Oscillator problem. 

By the Rabi Oscillator problem, we mean the two-level system driven by a
periodic field:
    du/dt = A(t) u

Lab frame (exact):
    A_lab(t) = -i [(ω₀/2)σz + Ω cos(ωt) σx]
             = -i [ω₀/2        Ω cos(ωt)]
                  [Ω cos(ωt)      -ω₀/2 ]

Rotating frame without RWA:

Rotating frame with RWA:
    A_rot = -i [(Δ/2)σz + (Ω/2)σx]
          = -i [Δ/2    Ω/2]
               [Ω/2   -Δ/2]
    where Δ = ω₀ - ω is the detuning.
"""


# =============================================================================
# Main experiment parameters
# =============================================================================

ω₀ = 1            # Atomic transition frequency
ω = 1             # Drive frequency (on resonance: ω = ω₀)
Ω = 0.01          # drive strength / Rabi frequenci
refinementFactor = 2

# For each s-value of the Filon method, run test for various numbers of timesteps
# (if timesteps are floats, will be rounded *up*)
filon_s_to_nsteps = (
    0 => refinementFactor .^ (2:16),
    #1 => refinementFactor .^ (2:8),
    #2 => refinementFactor .^ (2:8),
)

hermite_s_to_nsteps = (
    0 => refinementFactor .^ (2:16),
    #1 => refinementFactor .^ (2:8),
    #2 => refinementFactor .^ (2:8),
)

nsaves = 4 # Number of time points (after initial condition) to save to jld2 files
initialCondition = "excited" # "excited" or "ground"

period_lab = begin 
    Δ = ω₀ - ω              # Detuning
    Ω_eff = sqrt(Δ^2 + Ω^2) # Generalized Rabi frequency

    period_lab = 2pi / (ω₀/2)    # Lab period
    period_rabi = 2pi / Ω_eff    # Rabi period
    period_lab  # Final time
end

Tmax = 25.5 * period_lab

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
@everywhere using LinearAlgebra
@everywhere using Printf
@everywhere using OrdinaryDiffEqVerner
@everywhere include(srcdir("error_analysis.jl"))
@everywhere include(srcdir("rabiOscillator_helpers.jl"))

@everywhere const prefix = "rabiOscillatorConvergence"
@everywhere const outdir = datadir(prefix)

# ============================================================
# Set up functions for running simulations
# ============================================================

@everywhere function make_initial_condition(spec)
    if spec == "ground"
        return ComplexF64[1, 0]
    elseif spec == "excited"
        return ComplexF64[0, 1]
    else
        throw(ArgumentError("Invalid spec '$spec'. Options are 'ground' and 'excited'."))
    end
end

@everywhere function run_simulation(config)
    ψ₀ = make_initial_condition(config.initialCondition)
    Δ = config.ω₀ - config.ω

    # Set up hamiltonian
    if config.frame == :lab
        A_deriv_funcs = A_lab_funcs(ω₀=config.ω₀, ω=config.ω, Ω=config.Ω)
    elseif config.frame == :rot_rwa
        A_deriv_funcs = A_rot_funcs(Ω=config.Ω, Δ=Δ)
    else
        throw(ArgumentError("Invalid frame '$(config.frame)'. Options are ':lab' and ':rot_rwa'"))
    end

    t_saves = collect(range(0, config.Tmax, length=1+config.nsaves))

    # Set up Filon frequencies
    if config.method == :hermite
        filon_freqs = zeros(2) # No frequencies = Hermite
    elseif config.method == :filon
        if config.frame == :lab
            filon_freqs = -config.ω₀/2 .* Float64[1, -1]
        elseif config.frame == :rot_rwa
            filon_freqs = -Δ/2 .* Float64[1, -1] # TODO double check sign
        else
            throw(ArgumentError("Invalid frame '$(config.frame)'. Options are ':lab' and ':rot_rwa'"))
        end
    else
        throw(ArgumentError("Invalid method '$(config.method)'. Method must be either 'hermite' or 'filon'."))
    end

    # Run once to get compilation out of the way
    dummy_nsteps = 1
    t_elapsed = @elapsed history = filon_solve(
        A_deriv_funcs, ψ₀, filon_freqs, config.Tmax, dummy_nsteps, config.s,
    )
    # Run again, for real this time
    t_elapsed = @elapsed history = filon_solve(
        A_deriv_funcs, ψ₀, filon_freqs, config.Tmax, config.nsteps, config.s,
    )
    # Downsample history and store
    history = downsample_history(history, config.nsaves)

    return @strdict history t_elapsed t_saves
end

@everywhere function run_simulation_reference(config)
    u0 = make_initial_condition(config.initialCondition)
    t_saves = collect(range(0, config.Tmax, length=1+config.nsaves))
    tspan = (0, config.Tmax) # Not times to save it, but the time interval of the problem

    @unpack ω₀, ω, Ω = config
    Δ = ω₀ - ω
    function rhs_lab!(du, u, p, t)
        c = cos(ω * t)
        du[1] = -im * ((ω₀ / 2) * u[1] + Ω * c * u[2])
        du[2] = -im * (Ω * c * u[1] - (ω₀ / 2) * u[2])
        return nothing
    end

    function rhs_rot_rwa!(du, u, p, t)
        du[1] = -im * ((Δ / 2) * u[1] + (Ω / 2) * u[2])
        du[2] = -im * ((Ω / 2) * u[1] - (Δ / 2) * u[2])
        return nothing
    end

    # Set up hamiltonian
    if config.frame == :lab
        prob = ODEProblem(rhs_lab!, u0, tspan)
    elseif config.frame == :rot_rwa
        prob = ODEProblem(rhs_rot_rwa!, u0, tspan)
    else
        throw(ArgumentError("Invalid frame '$(config.frame)'. Options are ':lab' and ':rot_rwa'"))
    end

    # Techincally not a great timing since this may not be compiled yet
    t_elapsed = @elapsed sol = solve(
        prob, Vern9(); saveat=t_saves, reltol=1e-14, abstol=1e-14
    )
    history = reduce(hcat, sol.u)

    return @strdict history t_elapsed t_saves
end


@everywhere flush(stdout)

# ======================
# Collect data
# ======================

# Compute reference solutions using OrdinaryDiffEq.jl
reference_config_lab = (;frame=:lab, ω₀, ω, Ω, Tmax, initialCondition, nsaves)
reference_config_rot = (;frame=:rot_rwa, ω₀, ω, Ω, Tmax, initialCondition, nsaves)
produce_or_load(run_simulation_reference, reference_config_lab, outdir,
                filename = c -> savename(prefix, c, sort=false))
produce_or_load(run_simulation_reference, reference_config_rot, outdir,
                filename = c -> savename(prefix, c, sort=false))

configs = []

for frame in (:lab, :rot_rwa)
    for (s, nsteps_vec) in filon_s_to_nsteps
        for nsteps_maybe_float in nsteps_vec
            nsteps = round(Int, nsteps_maybe_float) 
            # Create config as a NamedTuple [`(; x, y, ...)` infers variable names from arguments]
            push!(
                configs,
                (; method=:filon, frame, ω₀, ω, Ω, s, Tmax, initialCondition,
                   nsaves, refinementFactor, nsteps)
            )
        end
    end
end


for frame in (:lab, :rot_rwa)
    for (s, nsteps_vec) in hermite_s_to_nsteps
        for nsteps_maybe_float in nsteps_vec
            nsteps = round(Int, nsteps_maybe_float) 
            # Create config as a NamedTuple [`(; x, y, ...)` infers variable names from arguments]
            push!(
                configs,
                (; method=:hermite, frame, ω₀, ω, Ω, s, Tmax, initialCondition,
                   nsaves, refinementFactor, nsteps)
            )
        end
    end
end

run_successes = pmap(configs) do config
    try
        print("frame = $(config.frame) | ")
        process_convergence_config(run_simulation, config, prefix, outdir)
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
        print("frame = $(config.frame) | ")
        process_convergence_config(run_simulation, config, prefix, outdir)
    catch ex
        @warn "Simulation failed" config exception=(ex, catch_backtrace())
        false
    end
end
flush(stdout)
