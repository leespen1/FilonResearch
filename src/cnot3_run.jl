"""
Simulation driver for the CNOT3 convergence experiment: builds the problem,
operators, and runs one of the three integrators for a single config.  Shared by
the data-collection script and the smoke test.

Assumes `FilonResearch`, `QuantumGateDesign`, and `LinearAlgebra.norm` are in
scope in the including module.
"""

using Statistics: mean, median, std

include(srcdir("cnot3_hoho_helpers.jl"))
include(srcdir("QuantumGateDesign_interface.jl"))

# Construct the initial condition from its spec and the QGD problem:
#   * "uniform" — the equal-weight superposition over all levels (one state
#     vector);
#   * "basis"   — the full essential computational basis (the N_tot × N_ess
#     matrix of gate states QuantumGateDesign propagates to evaluate a gate, its
#     `u0`/`v0` fields).
function make_initial_condition(spec, qgd_prob)
    n = qgd_prob.N_tot_levels
    if spec == "uniform"
        u0 = ones(ComplexF64, n)
        return u0 ./ norm(u0)
    elseif spec == "basis"
        return complex.(qgd_prob.u0, qgd_prob.v0)
    end
    throw(ArgumentError("unknown initial condition '$spec'; choose \"basis\" or \"uniform\""))
end

# Solve every column with `solve!(c, stats)` (which records per-step GMRES
# iteration counts into `stats`), returning the stacked history, the wall-clock
# solve time, and the per-step GMRES iteration counts pooled over all steps of
# all columns.  Used by the iterative (Filon-family) methods to expose how hard
# the linear solves work; the explicit QGD Hermite method has no GMRES and so
# returns an empty count vector.
function solve_columns_with_stats(solve!, cols, stack_histories)
    stats = FilonSolveStats()
    niters = Int[]
    solve_one = function (c)
        h = solve!(c, stats)          # solvers empty! `stats` at entry
        append!(niters, stats.gmres_niters)
        return h
    end
    t_elapsed = @elapsed history = stack_histories(map(solve_one, cols))
    return history, t_elapsed, niters
end

# Summarize the pooled per-step GMRES iteration counts.  Returns `missing` for
# every statistic when no GMRES was used (the explicit QGD Hermite method), so
# the result columns line up across all methods.
function gmres_summary(niters)
    isempty(niters) && return (; mean = missing, median = missing,
                               max = missing, std = missing)
    return (; mean = mean(niters), median = median(niters),
            max = maximum(niters), std = std(niters))
end

# Solve the ODE for one config and return a dictionary holding the saved history
# (complex, N × (nsaves+1), on the shared `range(0, Tmax, nsaves+1)` grid), the
# wall-clock time of the (post-compilation) solve, the saved times, summary
# statistics of the per-step GMRES iteration counts (for the iterative methods;
# `missing` for QGD Hermite, which has no linear solve), the git commit the run
# was produced on (and whether the working tree was dirty), and the config fields
# (so `collect_results` exposes the parameter columns directly).
#
# Errors are deliberately *not* recorded here: they are computed downstream
# against the Vern9 reference solution (see `cnot3_reference.jl`), so a run's
# accuracy never depends on which other runs happen to share its data directory.
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

    atol = config.gmresAtol
    rtol = config.gmresRtol

    niters = Int[]        # per-step GMRES counts; stays empty for QGD Hermite
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
        filon_solve_hardcoded(co, cols[1], freqs, config.Tmax / 2, 2,
                              config.s; save_final_only = true)            # warm-up
        history, t_elapsed, niters = solve_columns_with_stats(
            (c, stats) -> filon_solve_hardcoded(
                co, c, freqs, config.Tmax / nsteps, nsteps, config.s;
                save_every, stats, gmres_atol = atol, gmres_rtol = rtol),
            cols, stack_histories)
    elseif config.method == :controlled_filon
        co = qgd_to_controlled_filon_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        controlled_filon_solve(co, cols[1], freqs, config.Tmax / 2, 2,
                               config.s; save_final_only = true)           # warm-up
        history, t_elapsed, niters = solve_columns_with_stats(
            (c, stats) -> controlled_filon_solve(
                co, c, freqs, config.Tmax / nsteps, nsteps, config.s;
                save_every, stats, gmres_atol = atol, gmres_rtol = rtol),
            cols, stack_histories)
    elseif config.method == :controlled_hermite
        # The ω = 0 (Hermite) counterpart of the efficient controlled Filon
        # method: same full A(t) operator as :filon (carriers folded into the
        # controls), but each control matrix is applied only s+1 times per step.
        co = qgd_to_controlled_operator(qgd_prob, controls, pcof)
        efficient_controlled_hermite_solve(co, cols[1], config.Tmax / 2, 2,
                                           config.s; save_final_only = true) # warm-up
        history, t_elapsed, niters = solve_columns_with_stats(
            (c, stats) -> efficient_controlled_hermite_solve(
                co, c, config.Tmax / nsteps, nsteps, config.s;
                save_every, stats, gmres_atol = atol, gmres_rtol = rtol),
            cols, stack_histories)
    else
        throw(ArgumentError("Invalid method '$(config.method)'. Must be " *
            ":hermite, :filon, :controlled_filon, or :controlled_hermite."))
    end

    g = gmres_summary(niters)

    # Provenance: the commit this run was produced on, and whether the working
    # tree had uncommitted changes (gitdescribe appends a "-dirty" suffix).  The
    # dirty state is recorded, not warned about — a dirty tree is fine here.
    gitdesc = gitdescribe(projectdir(); warn = false)
    gitdirty = endswith(gitdesc, "-dirty")
    gitcommit = gitdirty ? chop(gitdesc; tail = length("-dirty")) : gitdesc

    output = Dict{String,Any}(
        "history"      => history,
        "t_elapsed"    => t_elapsed,
        "t_saves"      => t_saves,
        "gmres_mean"   => g.mean,
        "gmres_median" => g.median,
        "gmres_max"    => g.max,
        "gmres_std"    => g.std,
        "gitcommit"    => gitcommit,
        "gitdirty"     => gitdirty,
    )
    for (k, v) in pairs(config)
        output[string(k)] = v
    end
    return output
end
