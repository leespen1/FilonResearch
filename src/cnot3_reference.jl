"""
Vern9 reference solution for the CNOT3 convergence experiment.

Every run's error is measured against an independent, high-accuracy reference:
the same A(t) dynamics integrated with the adaptive Verner 9 Runge–Kutta method
at a tight tolerance (1e-15 by default).  The reference is cached on disk
(`datadir("cnot3_vern9ref")`), so the expensive solve happens only once per
`(frame, initialCondition, …)` and is shared by the summarize, plot, and paper
scripts.

Assumes `cnot3_run.jl` is already `include`d (for `cnot3_hoho_qgd_prob`,
`cnot3_hoho_controls_and_pcof`, `qgd_to_controlled_operator`, and
`make_initial_condition`) and that `FilonResearch` is in scope (for `Operator` /
`evaluate!`).
"""

using DrWatson
using OrdinaryDiffEqVerner
using LinearAlgebra: mul!, norm

# Integrate dψ/dt = A(t)ψ with Vern9, one column per initial state, saving on the
# shared `range(0, Tmax, nsaves+1)` grid.  Columns are stacked vertically to match
# the layout `run_simulation` produces for the "basis" initial condition.
function compute_vern9_reference(; frame, initialCondition, Nosc, Nguard, Tmax,
                                 nsaves, abstol = 1e-15, reltol = 1e-15)
    fr = Symbol(frame)
    qgd_prob = cnot3_hoho_qgd_prob(N_osc_levels = Nosc, N_guard_levels = Nguard,
                                   Tmax = Tmax, frame = fr)
    controls, pcof = cnot3_hoho_controls_and_pcof(frame = fr)
    co = qgd_to_controlled_operator(qgd_prob, controls, pcof)   # A(t) = Σ cₖ(t) Aₖ
    ic = make_initial_condition(initialCondition, qgd_prob)
    cols = ic isa AbstractVector ? (ic,) : collect(eachcol(ic))
    t_saves = collect(range(0, Tmax, length = 1 + nsaves))

    op = Operator(co, 0.0)                                      # reusable A(t) buffer
    function rhs!(du, u, p, t)
        evaluate!(op, co, t)                                   # refresh A(t) in place
        mul!(du, op, u)
        return nothing
    end

    histories = map(cols) do c
        u0 = ComplexF64.(Vector(c))
        # The lab frame's carriers force the adaptive solver to take many steps to
        # reach 1e-15, far past the default cap; raise maxiters and fail loudly if
        # the integration still stops short of Tmax rather than returning a
        # silently truncated (wrong-final-time) reference.
        sol = solve(ODEProblem(rhs!, u0, (0.0, Tmax)), Vern9();
                    abstol = abstol, reltol = reltol, saveat = t_saves,
                    maxiters = 20_000_000)
        length(sol.t) == length(t_saves) && isapprox(sol.t[end], Tmax; atol = 1e-9) ||
            error("Vern9 reference (frame=$frame, init=$initialCondition) stopped early at " *
                  "t=$(sol.t[end]) of $Tmax (hit the step cap); raise maxiters")
        reduce(hcat, (Vector{ComplexF64}(u) for u in sol.u))
    end
    return length(histories) == 1 ? histories[1] : reduce(vcat, histories)
end

"""
    vern9_reference(; frame, initialCondition, Nosc, Nguard, Tmax, nsaves,
                    abstol = 1e-15, reltol = 1e-15) -> Dict

Cached Vern9 reference for one configuration.  Returns a dictionary with

  * `"href"` — the full reference history on the `nsaves`-point save grid (same
    layout as a run's `history`), and
  * `"uref"` — the final-time reference state (`href[:, end]`).

Cached in a SEPARATE data dir (`datadir("cnot3_vern9ref")`, not under the run
prefix) so `collect_results` over the run data never picks these files up.
"""
function vern9_reference(; frame, initialCondition, Nosc, Nguard, Tmax, nsaves,
                         abstol = 1e-15, reltol = 1e-15)
    cfg = Dict("frame" => string(frame), "initialCondition" => string(initialCondition),
               "abstol" => abstol, "reltol" => reltol, "Nosc" => Nosc,
               "Nguard" => Nguard, "Tmax" => Tmax, "nsaves" => nsaves)
    data, _ = produce_or_load(cfg, datadir("cnot3_vern9ref");
                              prefix = "cnot3_vern9ref", tag = false) do _
        println("  computing Vern9 reference: frame=$frame init=$initialCondition ",
                "(this is the slow step)")
        href = compute_vern9_reference(; frame, initialCondition, Nosc, Nguard,
                                       Tmax, nsaves, abstol, reltol)
        Dict("href" => href, "uref" => href[:, end])
    end
    return data
end

# Note: the lightweight loader `load_vern9_reference` and `reference_errors` live
# in error_analysis.jl, which has no ODE-solver dependency, so the summarize/plot
# scripts can read cached references without loading this file's solver stack.
