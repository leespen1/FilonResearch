"""
Simulation driver for the CNOT3 convergence experiment: builds the problem,
operators, and runs one of the three integrators for a single config.  Shared by
the data-collection script and the smoke test.

Assumes `FilonResearch`, `QuantumGateDesign`, and `LinearAlgebra.norm` are in
scope in the including module.
"""

include(srcdir("cnot3_hoho_helpers.jl"))
include(srcdir("QuantumGateDesign_interface.jl"))

# Construct the initial condition vector from its spec and the system size.
function make_initial_condition(spec, n::Integer)
    if spec == "uniform" # Uniform superposition
        u0 = ones(ComplexF64, n)
        return u0 ./ norm(u0)
    elseif spec isa AbstractString # Basis vector "eN"
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

# Solve the ODE for one config, returning the saved history (complex,
# N × (nsaves+1), on the shared `range(0, Tmax, nsaves+1)` grid), the wall-clock
# time of the (post-compilation) solve, the saved times, and the config fields
# (so `collect_results` exposes the parameter columns directly).
function run_simulation(config)
    qgd_prob = cnot3_hoho_qgd_prob(
        N_osc_levels = config.nOscLevels,
        N_guard_levels = config.nGuardLevels,
        Tmax = config.Tmax,
    )
    N = qgd_prob.N_tot_levels
    initial_condition = make_initial_condition(config.initialCondition, N)
    controls, pcof = cnot3_hoho_controls_and_pcof()

    order = 2 * (config.s + 1)
    nsteps = config.nsteps
    save_every = div(nsteps, config.nsaves)
    t_saves = collect(range(0, config.Tmax, length = 1 + config.nsaves))

    if config.method == :hermite
        # Warm-up to compile before timing.
        eval_forward_complex_history(qgd_prob, controls, pcof, initial_condition;
                                     order, nsteps = config.nsaves, saveEveryNsteps = 1)
        t_elapsed = @elapsed history = eval_forward_complex_history(
            qgd_prob, controls, pcof, initial_condition;
            order, nsteps, saveEveryNsteps = save_every)
    elseif config.method == :filon
        co = qgd_to_controlled_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        filon_solve_hardcoded(co, initial_condition, freqs, config.Tmax / 2, 2,
                              config.s; save_final_only = true)            # warm-up
        t_elapsed = @elapsed history = filon_solve_hardcoded(
            co, initial_condition, freqs, config.Tmax / nsteps, nsteps, config.s; save_every)
    elseif config.method == :controlled_filon
        co = qgd_to_controlled_filon_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        controlled_filon_solve(co, initial_condition, freqs, config.Tmax / 2, 2,
                               config.s; save_final_only = true)           # warm-up
        t_elapsed = @elapsed history = controlled_filon_solve(
            co, initial_condition, freqs, config.Tmax / nsteps, nsteps, config.s; save_every)
    else
        throw(ArgumentError("Invalid method '$(config.method)'. " *
            "Must be :hermite, :filon, or :controlled_filon."))
    end

    output = @strdict history t_elapsed t_saves
    for (k, v) in pairs(config)
        output[string(k)] = v
    end
    return output
end
