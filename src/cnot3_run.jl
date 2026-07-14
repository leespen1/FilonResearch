"""
Simulation driver for the CNOT3 convergence experiment: builds the problem,
operators, and runs one of its integrators for a single config.  Shared by
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
# GMRES iteration counts (every method here is a FilonResearch GMRES solver), the
# git commit the run was produced on (and whether the working tree was dirty), and
# the config fields (so `collect_results` exposes the parameter columns directly).
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

    sv    = config.s
    # One carrier-grouped operator shared by every method: each control contributes
    # its two constant matrices once, paired with a `SumControl` over that control's
    # carriers.  This is the form Appendix B's efficient controlled-Filon method
    # needs (carriers folded into the per-matrix generator, each matrix applied once
    # per stage regardless of carrier count), and the Filon/Hermite methods consume
    # exactly the same operator — they evaluate the controls via `derivative`, which
    # folds the carriers — so the method comparison is over an identical operator.
    co    = qgd_to_efficient_controlled_filon_operator(qgd_prob, controls, pcof)
    freqs = qgd_ansatz_frequencies(qgd_prob)

    # Guard the carrier grouping: the operator must have one drift plus two matrices
    # per control (M⁺ₖ, M⁻ₖ), not two per control *per carrier*.  A revert to the
    # ungrouped adapter (`qgd_to_controlled_filon_operator`) would make this 1 +
    # 2·ncontrol·Nfreq and silently restore the per-carrier matvec blowup.
    @assert length(co) == 1 + 2 * length(controls) (
        "expected a carrier-grouped controlled operator with $(1 + 2*length(controls)) " *
        "matrices (one drift + M⁺/M⁻ per control); got $(length(co)).  The carriers of " *
        "each control must be gathered under a SumControl, not split across duplicated matrices.")

    # Each method is a `step(c, Δt, n, stats)` advancing one column with its solver
    # and recording GMRES stats.  Each `Naive*` / non-`Naive` pair is the same
    # method to ~1e-14; the non-Naive ("efficient") variants reorganize the
    # operator apply (each control matrix applied s+1 times) — a large win for the
    # ControlledFilon and Hermite families, while for plain Filon the drift is
    # diagonal so the efficient apply is not faster (both are offered regardless).
    # `:NaiveControlledFilon` is the lone exception that does *not* use the shared
    # `co`: its solver loops over controls directly without reaching into a
    # SumControl, so it needs the per-carrier-split operator built below.
    # Wrapping the solver in one closure makes the warm-up and the timed solve use
    # the identical keyword call site, so compilation stays out of the timed region.
    if config.method === :NaiveHermite
        step = (c, Δt, n, stats) -> hermite_solve_hardcoded(co, c, Δt, n, sv;
            save_final_only = sfo, save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol)
    elseif config.method === :Hermite
        step = (c, Δt, n, stats) -> efficient_controlled_hermite_solve(co, c, Δt, n, sv;
            save_final_only = sfo, save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol)
    elseif config.method === :NaiveFilon
        step = (c, Δt, n, stats) -> filon_solve_hardcoded(co, c, freqs, Δt, n, sv;
            save_final_only = sfo, save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol)
    elseif config.method === :Filon
        step = (c, Δt, n, stats) -> efficient_filon_solve(co, c, freqs, Δt, n, sv;
            save_final_only = sfo, save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol)
    elseif config.method === :NaiveControlledFilon
        # The naive solver loops over controls directly, so each carrier must be its
        # own bare CarrierControl: build the per-carrier-split operator here.
        co_cf = qgd_to_controlled_filon_operator(qgd_prob, controls, pcof)
        step = (c, Δt, n, stats) -> controlled_filon_solve(co_cf, c, freqs, Δt, n, sv;
            save_final_only = sfo, save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol)
    elseif config.method === :ControlledFilon
        step = (c, Δt, n, stats) -> efficient_controlled_filon_solve(co, c, freqs, Δt, n, sv;
            save_final_only = sfo, save_every = div(n, config.nsaves), stats, gmres_atol = atol, gmres_rtol = rtol)
    elseif config.method === :HermiteQGD
        # QuantumGateDesign's own Hermite time-stepper at order 2(s+1) — a baseline
        # the hand-rolled :Hermite method is compared against.  QGD steps internally
        # (no FilonResearch per-step GMRES stats, so `stats` stays empty), and its
        # GMRES tolerance is the one baked into `qgd_prob` rather than `atol`/`rtol`.
        order = 2 * (sv + 1)
        step = (c, Δt, n, stats) -> begin
            h = eval_forward_complex_history(qgd_prob, controls, pcof, c;
                order, nsteps = n, saveEveryNsteps = div(n, config.nsaves))
            sfo ? h[:, end:end] : h
        end
    else
        throw(ArgumentError("Invalid method '$(config.method)'.  Choose from :NaiveHermite, " *
            ":Hermite, :NaiveFilon, :Filon, :NaiveControlledFilon, :ControlledFilon, :HermiteQGD."))
    end
    do_solve = (n) -> solve_columns((c, stats) -> step(c, config.Tmax / n, n, stats),
                                    cols, stack_histories)

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
