# Physics verification of the CNOT3 frame transformations on a small system.
# All three frames (rwa / norwa / lab) describe the same physical pulse
# f_k(t) = 2 Re[α_k(t) e^{i ω_k t}], so:
#
#   1. the norwa complex amplitude must satisfy γ = α + conj(α) e^{−2iωt} and
#      the lab pulse must satisfy p_lab = 2 Re[α e^{iωt}]  (checked pointwise,
#      with derivatives against finite differences);
#   2. the norwa and lab *solutions* must agree exactly up to the diagonal
#      rotation phase e^{iWT}, W = Σₖ ωₖ nₖ;
#   3. the rwa solution must differ from norwa by a small but nonzero amount
#      (the physical RWA error).
#
# Not part of the experiment; run by hand:
#   julia --project=. scripts/cnot3/verify_frames.jl

using DrWatson
@quickactivate "FilonExperiments"

using FilonResearch
using QuantumGateDesign
using LinearAlgebra
using Printf
using Random

include(srcdir("cnot3_run.jl"))

Random.seed!(1234)

# --- control-level identities (full-size controls, cheap) --------------------
controls_rwa, pcof_rwa = cnot3_hoho_controls_and_pcof()
ctrl = Dict(fr => cnot3_hoho_controls_and_pcof(frame = fr) for fr in CNOT3_FRAMES)
ωs = cnot3_hoho_rotation_freqs()

maxd_norwa = maxd_lab = 0.0
for t in 550.0 .* rand(20), k in 1:3
    slice(cs, p) = QuantumGateDesign.get_control_vector_slice(p, cs, k)
    αval(cs, p) = QuantumGateDesign.eval_p(cs[k], t, slice(cs, p)) +
                  im * QuantumGateDesign.eval_q(cs[k], t, slice(cs, p))
    α = αval(controls_rwa, pcof_rwa)
    γ = αval(ctrl[:norwa]...)
    f = QuantumGateDesign.eval_p(ctrl[:lab][1][k], t, slice(ctrl[:lab]...))
    global maxd_norwa = max(maxd_norwa, abs(γ - (α + conj(α) * exp(-2im * ωs[k] * t))))
    global maxd_lab = max(maxd_lab, abs(f - 2 * real(α * exp(im * ωs[k] * t))))
end
@printf "norwa amplitude identity max err: %.3e\n" maxd_norwa
@printf "lab pulse identity max err:       %.3e\n" maxd_lab
@assert maxd_norwa < 1e-12 "norwa controls do not match α + conj(α)e^{−2iωt}"
@assert maxd_lab < 1e-12 "lab controls do not match 2Re[αe^{iωt}]"

# Derivatives of the transformed controls against central finite differences.
# Carriers are ~50 rad/ns, so the FD truncation floor is ~(ωh)²/6 ≈ 4e-6.
fd_err = 0.0
for fr in (:norwa, :lab), k in 1:3, t in 100.0 .+ 350.0 .* rand(5), n in 1:3
    controls, pcof = ctrl[fr]
    pc = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
    h = 1e-4
    for evalf in (eval_p_derivative, eval_q_derivative)
        d = evalf(controls[k], t, pc, n)
        fd = (evalf(controls[k], t + h, pc, n - 1) -
              evalf(controls[k], t - h, pc, n - 1)) / 2h
        global fd_err = max(fd_err, abs(d - fd) / max(abs(d), 1.0))
    end
end
@printf "control derivative FD mismatch:   %.3e\n" fd_err
@assert fd_err < 1e-3 "transformed-control derivatives disagree with finite differences"
println("PASS: control-level frame identities.\n")

# --- solution-level frame equivalence (small system) -------------------------
Tmax = 50.0
prob = Dict(fr => cnot3_hoho_qgd_prob(; N_osc_levels = 3, N_guard_levels = 1,
                                      Tmax, frame = fr)
            for fr in CNOT3_FRAMES)
N = prob[:rwa].N_tot_levels
ic = ones(ComplexF64, N); ic ./= norm(ic)

# Rotation generator W = Σₖ ωₖ nₖ = diag(lab drift) − diag(rotating drift).
W = diag(Matrix(prob[:lab].system_sym)) .- diag(Matrix(prob[:rwa].system_sym))

# Converged rotating-frame references via QGD Hermite (order 6).
hsolve(fr, nsteps) = eval_forward_complex_history(
    prob[fr], ctrl[fr]..., ic; order = 6, nsteps, saveEveryNsteps = nsteps)[:, end]
ref = Dict(fr => hsolve(fr, 2^15) for fr in (:rwa, :norwa))
for fr in (:rwa, :norwa)
    selfconv = norm(hsolve(fr, 2^14) - ref[fr])
    @printf "%-6s Hermite self-convergence (2^14 vs 2^15): %.3e\n" fr selfconv
    @assert selfconv < 1e-10 "$fr Hermite reference is not converged"
end

# The lab frame oscillates too fast for Hermite at these step counts; solve it
# with controlled-Filon (which factors the carriers) and map back with e^{iWT}.
co_lab = qgd_to_controlled_filon_operator(prob[:lab], ctrl[:lab]...)
freqs_lab = qgd_ansatz_frequencies(prob[:lab])
nsteps = 2^12
hist = controlled_filon_solve(co_lab, ic, freqs_lab, Tmax / nsteps, nsteps, 2;
                              save_final_only = true)
ψ_lab_rotated = exp.(im .* W .* Tmax) .* hist[:, end]

frame_equiv_err = norm(ψ_lab_rotated - ref[:norwa])
rwa_err = norm(ref[:rwa] - ref[:norwa])
@printf "\n|e^{iWT}ψ_lab(T) − ψ_norwa(T)| = %.3e   (numerical error only)\n" frame_equiv_err
@printf "|ψ_rwa(T) − ψ_norwa(T)|        = %.3e   (physical RWA error)\n" rwa_err
@assert frame_equiv_err < 1e-6 "lab and norwa frames disagree beyond solver error"
@assert 1e-6 < rwa_err < 1e-1 "RWA error has unexpected magnitude"
println("PASS: frames produce the same physical solution (rwa differs only by the RWA).")
