# Scratch verification of the QGD → ControlledOperator adapters on a small CNOT3
# system.  Not part of the experiment; run by hand:
#   julia --project=. scripts/cnot3/verify_adapters.jl

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

# --- small CNOT3 system (cheap) ----------------------------------------------
Tmax = 550.0
qgd_prob = cnot3_hoho_qgd_prob(N_osc_levels = 3, N_guard_levels = 1, Tmax = Tmax)
controls, pcof = cnot3_hoho_controls_and_pcof()
N = qgd_prob.N_tot_levels
println("System size N = ", N)

co_filon  = qgd_to_controlled_operator(qgd_prob, controls, pcof)
co_cfilon = qgd_to_controlled_filon_operator(qgd_prob, controls, pcof)
co_eff    = qgd_to_efficient_controlled_filon_operator(qgd_prob, controls, pcof)
println("Filon operator:                      ", co_filon)
println("Controlled-Filon operator (split):   ", co_cfilon)
println("Controlled-Filon operator (grouped): ", co_eff)

# --- structural contract of the two controlled-Filon adapters -----------------
# The grouped adapter (fed to efficient_controlled_filon_solve) gathers each
# control's carriers under one SumControl, so it has one drift + M⁺/M⁻ per control.
# The split adapter (fed to the naive controlled_filon_solve) keeps every carrier
# as its own bare CarrierControl, so it has two matrices per control *per carrier*.
ncontrol = length(controls)
Nfreq = length(first(controls).carrier_frequencies)
@assert length(co_eff) == 1 + 2 * ncontrol
@assert length(co_cfilon) == 1 + 2 * ncontrol * Nfreq
@assert all(c isa FilonResearch.SumControl for c in co_eff.controls[2:end])
@assert all(length(FilonResearch.components(c)) == Nfreq for c in co_eff.controls[2:end])
println("PASS: grouped adapter has $(length(co_eff)) matrices (1 + 2·ncontrol), each ",
        "control a SumControl over $Nfreq carriers; split adapter has $(length(co_cfilon)).")

# --- A(t) equivalence at random times, derivative orders 0,1,2 ---------------
println("\nA(t) equivalence check (max abs difference):")
maxdiff = 0.0
for t in Tmax .* rand(8)
    for (dlabel, d) in (("A  ", Derivative{0}()), ("A' ", Derivative{1}()), ("A''", Derivative{2}()))
        Af  = materialize(evaluate(co_filon,  t, d))
        Acf = materialize(evaluate(co_cfilon, t, d))
        diff = maximum(abs.(Af .- Acf))
        global maxdiff = max(maxdiff, diff)
        @printf "  t=%8.3f  %s  diff = %.3e\n" t dlabel diff
    end
end
println("Overall max A(t)-derivative difference: ", maxdiff)
@assert maxdiff < 1e-9 "controlled-Filon operator does not match Filon operator"
println("PASS: operators reproduce the same A(t).\n")

# --- solver agreement: both Filon variants vs QGD Hermite --------------------
ic = ones(ComplexF64, N); ic ./= norm(ic)
freqs = qgd_ansatz_frequencies(qgd_prob)
nsaves = 4
println("Convergence toward the QGD Hermite reference (final-time 2-norm):")
nsteps_ref = 2^14
href = eval_forward_complex_history(qgd_prob, controls, pcof, ic;
                                    order = 6, nsteps = nsteps_ref,
                                    saveEveryNsteps = nsteps_ref ÷ nsaves)
uref = href[:, end]

for s in (0, 1, 2)
    println("  s = $s  (order $(2*(s+1)))")
    for nsteps in (2^8, 2^10, 2^12)
        se = nsteps ÷ nsaves
        hf  = filon_solve_hardcoded(co_filon,  ic, freqs, Tmax/nsteps, nsteps, s; save_every = se)
        hcf = controlled_filon_solve(co_cfilon, ic, freqs, Tmax/nsteps, nsteps, s; save_every = se)
        @printf "    nsteps=%-6d  filon=%.3e  cfilon=%.3e\n" nsteps norm(hf[:,end].-uref) norm(hcf[:,end].-uref)
    end
end
println("\nDone.")
