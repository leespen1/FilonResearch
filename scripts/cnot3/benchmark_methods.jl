# Wall-clock benchmark of the Filon/Hermite family on the CNOT3 problem, plus
# QuantumGateDesign's own Hermite time-stepper for reference.  This is NOT a
# convergence study: every method is run at the SAME (moderate) number of steps.
#
# The simulation runs over T = 100 ns rather than the full 550 ns gate duration.
# The optimized pulse is still defined over the full 550 ns (so it is not
# "squeezed"; see cnot3_hoho_rwa_controls_and_pcof) — we simply integrate a
# sub-interval.  The shorter window holds ~5.5× fewer carrier oscillations, which
# lets even the low-order Hermite methods (which must RESOLVE those carriers,
# unlike Filon, which folds them into the ansatz) reach their asymptotic region
# at a moderate step count.  The reported final-state error (vs a fine high-order
# reference) shows the accuracy regime each method is in.  The shorter window
# also gives a smaller Δt, well above any coarse-step instability.
#
# Five methods, at orders 2/4/6 (s = 0/1/2):
#   * QGD Hermite          — QuantumGateDesign.eval_forward (order 2(s+1))
#   * Hermite  (regular)   — hermite_solve_hardcoded            on the full A(t)
#   * Filon    (regular)   — filon_solve_hardcoded (efficient ordering) on A(t)
#   * Hermite  (controlled)— efficient_controlled_hermite_solve  on the full A(t)
#   * Filon    (controlled)— efficient_controlled_filon_solve    on the carrier-split A(t)
#
# Run by hand:
#   julia --project=. scripts/cnot3/benchmark_methods.jl

using DrWatson
@quickactivate "FilonExperiments"

using FilonResearch
using QuantumGateDesign
using LinearAlgebra
using Printf
using Random

include(srcdir("cnot3_hoho_helpers.jl"))
include(srcdir("QuantumGateDesign_interface.jl"))

Random.seed!(1234)

# --- benchmark settings ------------------------------------------------------
# NSTEPS is shared by every method.  2^12 is moderate and comfortably asymptotic
# for CNOT3 (well above the ~2^10 instability floor); NSAMPLES timing repeats are
# reduced to their minimum (the standard robust estimator, least GC-polluted).
const NSTEPS   = 2^12
const NSAMPLES = 5

# Reference: a fine, high-order QGD Hermite solve (4× finer than NSTEPS).
const REF_ORDER  = 6
const REF_NSTEPS = 2^14

# --- CNOT3 system ------------------------------------------------------------
# Integrate a 100 ns sub-window of the 550 ns gate (the pulse itself spans the
# full 550 ns); fewer carrier oscillations let low-order Hermite go asymptotic.
Tmax = 100.0
qgd_prob = cnot3_hoho_qgd_prob(N_osc_levels = 3, N_guard_levels = 1, Tmax = Tmax)
controls, pcof = cnot3_hoho_controls_and_pcof()
N = qgd_prob.N_tot_levels
Δt = Tmax / NSTEPS

ic = ones(ComplexF64, N); ic ./= norm(ic)
freqs = qgd_ansatz_frequencies(qgd_prob)

# A(t) = Σₖ cₖ(t) Aₖ.  The Hermite/regular-Filon methods use the full operator
# (carriers folded into the controls); controlled Filon uses the carrier-split,
# matrix-grouped operator.
co_ctrl   = qgd_to_controlled_operator(qgd_prob, controls, pcof)
co_cfilon = qgd_to_efficient_controlled_filon_operator(qgd_prob, controls, pcof)

println("CNOT3 benchmark:  N = $N  (subsystems 3×3×3),  ",
        "ncontrol = $(length(controls)),  Nfreq/control = $(length(controls[1].carrier_frequencies))")
println("  T = $Tmax,  NSTEPS = $NSTEPS  (Δt = $(round(Δt; sigdigits = 4)))")
println("  matrices:  full A(t) = $(length(co_ctrl.matrices)),  carrier-split = $(length(co_cfilon.matrices))\n")

# Warm up once, then report the minimum of NSAMPLES wall-clock times.
function bench(f)
    f()
    return minimum(@elapsed(f()) for _ in 1:NSAMPLES)
end

# Fine reference final state (computed once).
uref = eval_forward_complex_history(qgd_prob, controls, pcof, ic;
                                    order = REF_ORDER, nsteps = REF_NSTEPS,
                                    saveEveryNsteps = REF_NSTEPS)[:, end]

# Mean GMRES iterations per step from a FilonSolveStats collector (empty ⇒ "—",
# e.g. QGD, which does not expose its per-step GMRES counts through this API).
mean_gmres(::Nothing) = "—"
mean_gmres(st) = isempty(st.gmres_niters) ? "—" : @sprintf("%.1f", sum(st.gmres_niters) / length(st.gmres_niters))

for s in (0, 1, 2)
    qorder = 2 * (s + 1)
    # One stats collector per FilonResearch solver; refreshed on each call (the
    # solvers `empty!` it), so after a run it holds that run's per-step counts.
    sH = FilonSolveStats(); sF = FilonSolveStats(); sHc = FilonSolveStats(); sFc = FilonSolveStats()
    methods = (
        ("QGD Hermite",          () -> eval_forward_complex_history(qgd_prob, controls, pcof, ic;
                                          order = qorder, nsteps = NSTEPS, saveEveryNsteps = NSTEPS)[:, end], nothing),
        ("Hermite (regular)",    () -> hermite_solve_hardcoded(co_ctrl, ic, Δt, NSTEPS, s; save_final_only = true, stats = sH), sH),
        ("Filon (regular)",      () -> filon_solve_hardcoded(co_ctrl, ic, freqs, Δt, NSTEPS, s;
                                          save_final_only = true, efficient = true, stats = sF), sF),
        ("Hermite (controlled)", () -> efficient_controlled_hermite_solve(co_ctrl, ic, Δt, NSTEPS, s; save_final_only = true, stats = sHc), sHc),
        ("Filon (controlled)",   () -> efficient_controlled_filon_solve(co_cfilon, ic, freqs, Δt, NSTEPS, s; save_final_only = true, stats = sFc), sFc),
    )
    @printf("order %d  (s = %d)\n", qorder, s)
    println("  ", rpad("method", 22), lpad("time", 11), lpad("per step", 16), lpad("gmres/step", 12), lpad("final err", 13))
    for (name, f, st) in methods
        u = f()
        t = bench(f)
        row = @sprintf("%7.1f ms  %9.4f ms/step  %10s  %10.2e",
                       t * 1e3, t / NSTEPS * 1e3, mean_gmres(st), norm(u .- uref))
        println("  ", rpad(name, 22), row)
    end
    println()
end

println("Done.")
