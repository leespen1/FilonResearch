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
# iteration counts into `stats`), returning the stacked history and the per-step
# GMRES iteration counts pooled over all steps of all columns.  Used by the
# iterative (Filon-family) methods to expose how hard the linear solves work; the
# explicit QGD Hermite method has no GMRES and so returns an empty count vector.
# Timing is handled by the caller (run_simulation), which repeats and averages.
function solve_columns(solve!, cols, stack_histories)
    stats = FilonSolveStats()
    niters = Int[]
    solve_one = function (c)
        h = solve!(c, stats)          # solvers empty! `stats` at entry
        append!(niters, stats.gmres_niters)
        return h
    end
    history = stack_histories(map(solve_one, cols))
    return history, niters
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
# (complex; by default the full N × (nsaves+1) grid, or just the N × 1 final
# state when `saveFinalOnly = true`), the
# wall-clock solve time averaged over `config.nRuns` repetitions (compilation
# excluded — see below), the saved times, summary statistics of the per-step
# GMRES iteration counts (for the iterative methods; `missing` for QGD Hermite,
# which has no linear solve), the git commit the run was produced on (and whether
# the working tree was dirty), and the config fields (so `collect_results`
# exposes the parameter columns directly).
#
# Timing never includes compilation: each method's `do_solve(n)` closure is run
# once at a tiny step count (nsaves) as a warm-up before the timed region, which
# compiles the exact same function and keyword call site the full-length timed
# solve uses (a different keyword set would compile separately, inside timing).
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
    # `saveFinalOnly = true` keeps only the final state instead of the full
    # N × (nsaves+1) save grid.  The grid is the default (the l2-integral error
    # needs it); final-only is cheaper when only the final-time error matters.
    sfo = config.saveFinalOnly
    t_saves = sfo ? [config.Tmax] : collect(range(0, config.Tmax, length = 1 + config.nsaves))

    # A single state vector solves to one history; the "basis" matrix solves each
    # column with the same single-state solver and stacks the per-column
    # histories vertically (the final-time 2-norm of the stacked column is then
    # the Frobenius / gate-level error).  Per-column results are coerced to
    # matrices first, so the final-only case (a vector per column) stacks into an
    # (N·ncols) × 1 matrix, uniform with the full-history (N·ncols) × (nsaves+1).
    cols = initial_condition isa AbstractVector ? (initial_condition,) :
        collect(eachcol(initial_condition))
    to_mat(h) = h isa AbstractVector ? reshape(h, :, 1) : h
    stack_histories(hs) = length(hs) == 1 ? to_mat(hs[1]) : reduce(vcat, map(to_mat, hs))

    atol = config.gmresAtol
    rtol = config.gmresRtol

    # Each branch defines a single closure `do_solve(n)` that solves the problem
    # with `n` steps and returns (history, niters).  Running it once at the
    # smallest valid step count (nsaves) as a warm-up compiles the *exact* code
    # path — same function, same keyword call site — that the full-length timed
    # solve then uses, so compilation never lands inside the timed region.
    if config.method == :hermite
        do_solve = function (n)
            # saveEveryNsteps = n saves only the endpoints; keep the final column.
            se = sfo ? n : div(n, config.nsaves)
            hist = stack_histories(map(cols) do c
                h = eval_forward_complex_history(qgd_prob, controls, pcof, c;
                        order, nsteps = n, saveEveryNsteps = se)
                sfo ? h[:, end] : h
            end)
            return hist, Int[]                  # QGD Hermite has no GMRES
        end
    elseif config.method == :filon
        co = qgd_to_controlled_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        do_solve = (n) -> solve_columns(
            (c, stats) -> filon_solve_hardcoded(
                co, c, freqs, config.Tmax / n, n, config.s; save_final_only = sfo,
                save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol),
            cols, stack_histories)
    elseif config.method == :controlled_filon
        co = qgd_to_controlled_filon_operator(qgd_prob, controls, pcof)
        freqs = qgd_ansatz_frequencies(qgd_prob)
        do_solve = (n) -> solve_columns(
            (c, stats) -> controlled_filon_solve(
                co, c, freqs, config.Tmax / n, n, config.s; save_final_only = sfo,
                save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol),
            cols, stack_histories)
    elseif config.method == :controlled_hermite
        # The ω = 0 (Hermite) counterpart of the efficient controlled Filon
        # method: same full A(t) operator as :filon (carriers folded into the
        # controls), but each control matrix is applied only s+1 times per step.
        co = qgd_to_controlled_operator(qgd_prob, controls, pcof)
        do_solve = (n) -> solve_columns(
            (c, stats) -> efficient_controlled_hermite_solve(
                co, c, config.Tmax / n, n, config.s; save_final_only = sfo,
                save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol),
            cols, stack_histories)
    else
        throw(ArgumentError("Invalid method '$(config.method)'. Must be " *
            ":hermite, :filon, :controlled_filon, or :controlled_hermite."))
    end

    # Warm up (compile) outside the timed region, then time nRuns full-length
    # repetitions and average.  The solve is deterministic, so history and the
    # GMRES counts are identical across repetitions — only the wall-clock time is
    # averaged (to suppress measurement noise); keep the last repetition's data.
    do_solve(config.nsaves)
    local history, niters
    total_time = 0.0
    for _ in 1:config.nRuns
        total_time += @elapsed ((history, niters) = do_solve(nsteps))
    end
    t_elapsed = total_time / config.nRuns

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
