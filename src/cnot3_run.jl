"""
Simulation driver for the CNOT3 convergence experiment: builds the problem,
operators, and runs one of the three integrators for a single config.  Shared by
the data-collection script and the smoke test.

Assumes `FilonResearch`, `QuantumGateDesign`, and `LinearAlgebra.norm` are in
scope in the including module.
"""

include(srcdir("cnot3_hoho_helpers.jl"))
include(srcdir("QuantumGateDesign_interface.jl"))

# Construct the initial condition from its spec and the QGD problem.  Returns a
# single state vector for "uniform"/"eN", or — for "basis" — the N_tot × N_ess
# matrix of essential computational basis states that QuantumGateDesign
# propagates to evaluate a gate (its `u0`/`v0` fields).
function make_initial_condition(spec, qgd_prob)
    n = qgd_prob.N_tot_levels
    if spec == "uniform" # Uniform superposition
        u0 = ones(ComplexF64, n)
        return u0 ./ norm(u0)
    elseif spec == "basis" # Full essential computational basis (the gate states)
        return complex.(qgd_prob.u0, qgd_prob.v0)
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
    frame = Symbol(config.frame)
    qgd_prob = cnot3_hoho_qgd_prob(;
        N_osc_levels = config.nOscLevels,
        N_guard_levels = config.nGuardLevels,
        Tmax = config.Tmax,
        frame,
    )
    initial_condition = make_initial_condition(config.initialCondition, qgd_prob)
    controls, pcof = cnot3_hoho_controls_and_pcof(; frame)

    order = 2 * (config.s + 1)
    nsteps = config.nsteps
    save_every = div(nsteps, config.nsaves)
    t_saves = collect(range(0, config.Tmax, length = 1 + config.nsaves))

    # A single state vector solves to one N × (nsaves+1) history; the "basis"
    # matrix solves each column with the same single-state solver and stacks the
    # histories vertically into one (N·ncols) × (nsaves+1) array.  The final-time
    # 2-norm of a stacked column is then the Frobenius (gate-level) error, and
    # the downstream error analysis (which treats columns as state vectors and
    # time as the second axis) needs no changes.
    cols = initial_condition isa AbstractVector ? (initial_condition,) :
        collect(eachcol(initial_condition))
    stack_histories(hs) = length(hs) == 1 ? hs[1] : reduce(vcat, hs)

    if config.method == :hermite
        solve_one = c -> eval_forward_complex_history(
            qgd_prob, controls, pcof, c; order, nsteps, saveEveryNsteps = save_every)
        # Warm-up to compile before timing.
        eval_forward_complex_history(qgd_prob, controls, pcof, cols[1];
                                     order, nsteps = config.nsaves, saveEveryNsteps = 1)
        t_elapsed = @elapsed history = stack_histories(map(solve_one, cols))
    elseif config.method == :filon
        co = qgd_to_controlled_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        solve_one = c -> filon_solve_hardcoded(
            co, c, freqs, config.Tmax / nsteps, nsteps, config.s; save_every)
        filon_solve_hardcoded(co, cols[1], freqs, config.Tmax / 2, 2,
                              config.s; save_final_only = true)            # warm-up
        t_elapsed = @elapsed history = stack_histories(map(solve_one, cols))
    elseif config.method == :controlled_filon
        co = qgd_to_controlled_filon_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        solve_one = c -> controlled_filon_solve(
            co, c, freqs, config.Tmax / nsteps, nsteps, config.s; save_every)
        controlled_filon_solve(co, cols[1], freqs, config.Tmax / 2, 2,
                               config.s; save_final_only = true)           # warm-up
        t_elapsed = @elapsed history = stack_histories(map(solve_one, cols))
    elseif config.method == :controlled_hermite
        # The ω = 0 (Hermite) counterpart of the efficient controlled Filon
        # method: same full A(t) operator as :filon (carriers folded into the
        # controls), but each control matrix is applied only s+1 times per step.
        co = qgd_to_controlled_operator(qgd_prob, controls, pcof)
        solve_one = c -> efficient_controlled_hermite_solve(
            co, c, config.Tmax / nsteps, nsteps, config.s; save_every)
        efficient_controlled_hermite_solve(co, cols[1], config.Tmax / 2, 2,
                                           config.s; save_final_only = true) # warm-up
        t_elapsed = @elapsed history = stack_histories(map(solve_one, cols))
    else
        throw(ArgumentError("Invalid method '$(config.method)'. Must be " *
            ":hermite, :filon, :controlled_filon, or :controlled_hermite."))
    end

    output = @strdict history t_elapsed t_saves
    for (k, v) in pairs(config)
        output[string(k)] = v
    end
    return output
end
