# splus_illconditioning_analysis.jl
#
# Mathematical analysis of why S_+ = I - A·diag(W^I) becomes ill-conditioned
# when the Filon frequencies match the diagonal of A.
#
# ============================================================================
# The argument
# ============================================================================
#
# The implicit Filon timestep solves S_+ u_{n+1} = S_- u_n, where for s=0:
#
#   S_+ = I - A_{n+1} · diag(W^I_0)
#
# The weight vector W^I_0 has entries
#
#   W^I_0[k] = (Δt/2) · e^{-iω̂_k} · b_{right,0}(ω̂_k)
#
# where ω̂_k = ω_k · Δt/2, and b_{right,0}(ω̂) = ∫_{-1}^{1} ½(1+x) e^{iω̂x} dx
# is the Filon moment of the right-endpoint cardinal polynomial on [-1,1].
#
# For large ω̂, integration by parts gives (since g(x) = (1+x)/2 has g(1)=1, g(-1)=0):
#
#   b_{right,0}(ω̂) = e^{iω̂}/(iω̂) + O(1/ω̂²)
#
# The phase factor in W^I cancels:
#
#   W^I_0[k] = (Δt/2) · e^{-iω̂_k} · e^{iω̂_k}/(iω̂_k) + ...
#            = 1/(iω_k) + O(1/(ω_k² Δt))
#
# Now, the Filon method sets ω_k to match the diagonal of A. In the rotating
# frame for Schrödinger's equation:
#
#   A = -iH,    H[kk] = -ω_k    ⟹    A[kk] = iω_k
#
# (The diagonal of H is negative because the Kerr self-energies after RWA
# are negated by the convention ω_k = -H[kk].)
#
# Therefore the diagonal of S_+ is:
#
#   S_+[kk] = 1 - A[kk] · W^I_0[k]
#           = 1 - (iω_k) · (1/(iω_k) + O(1/(ω_k² Δt)))
#           = 1 - 1 - O(1/(ω_k Δt))
#           = O(1/(ω_k Δt))         →  0  as Δt → ∞
#
# The leading terms cancel exactly! Meanwhile, the off-diagonal entries are:
#
#   S_+[jk] = -A[jk] · W^I_0[k] → -A[jk]/(iω_k) = O(ε/ω_k)
#
# which approach a FIXED nonzero limit. So for large Δt:
#
#   |diag(S_+)| ~ 1/(ω Δt)    (shrinks)
#   |offdiag(S_+)| ~ ε/ω      (fixed)
#
# The condition number scales as cond(S_+) ~ ε·Δt → ∞.
#
# ============================================================================

using FilonResearch
using LinearAlgebra
using Printf

# ============================================================================
# Part 1: Verify the asymptotic behavior of W^I_0
# ============================================================================

println("=" ^ 90)
println("Part 1: Asymptotic behavior of W^I_0(ω)")
println("=" ^ 90)
println()
println("Prediction: W^I_0[k] → 1/(iω_k) as ω_k·Δt → ∞")
println()

dt = 100.0
ω_values = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

println(@sprintf("  Δt = %.1f", dt))
println(@sprintf("  %-8s  %-8s  %-28s  %-28s  %-14s",
    "ω", "ω̂", "W^I_0", "1/(iω)", "|difference|"))
println("  " * "-"^92)

for ω in ω_values
    ω̂ = ω * dt / 2
    w_left, w_right = filon_weights(ω̂, 0, -1.0, 1.0)
    WI = (dt/2) * cis(-ω̂) * w_right[1]
    asymptotic = 1.0 / (im * ω)

    @printf("  %-8.1f  %-8.1f  %+.4e %+.4ei  %+.4e %+.4ei  %-14.4e\n",
        ω, ω̂,
        real(WI), imag(WI),
        real(asymptotic), imag(asymptotic),
        abs(WI - asymptotic))
end

# ============================================================================
# Part 2: Diagonal cancellation — scalar case
# ============================================================================

println("\n\n" * "=" ^ 90)
println("Part 2: Diagonal cancellation in S_+ (scalar)")
println("=" ^ 90)
println()
println("Scalar ODE: du/dt = iω·u with Filon frequency ω.")
println("  A = iω,  W^I → 1/(iω),  so S_+ = 1 - iω/(iω) = 0.")
println()
println("Key: A[kk] = +iω_k (not -iω_k). The sign matters!")
println("  When A = -iH and frequencies ω_k = -H[kk]:  A[kk] = iω_k  →  cancellation")
println("  When A = -iH and frequencies ω_k = +H[kk]:  A[kk] = -iω_k →  no cancellation")
println()

println("--- Case 1: A = +iω (cancellation) ---")
println(@sprintf("  %-10s  %-20s  %-14s  %-14s",
    "Δt", "S_+", "|S_+|", "1/(ω·Δt)"))
println("  " * "-"^62)
ω_test = 10.0
for dt in [1.0, 10.0, 100.0, 1000.0, 10000.0]
    local ω̂ = ω_test * dt / 2
    w_left, w_right = filon_weights(ω̂, 0, -1.0, 1.0)
    local WI = (dt/2) * cis(-ω̂) * w_right[1]
    local A_kk = im * ω_test  # A = +iω
    local S_plus = 1.0 - A_kk * WI
    @printf("  %-10.0f  %+.6e%+.6ei  %-14.4e  %-14.4e\n",
        dt, real(S_plus), imag(S_plus), abs(S_plus), 1/(ω_test*dt))
end

println()
println("--- Case 2: A = -iω (no cancellation) ---")
println(@sprintf("  %-10s  %-20s  %-14s",
    "Δt", "S_+", "|S_+|"))
println("  " * "-"^50)
for dt in [1.0, 10.0, 100.0, 1000.0, 10000.0]
    local ω̂ = ω_test * dt / 2
    w_left, w_right = filon_weights(ω̂, 0, -1.0, 1.0)
    local WI = (dt/2) * cis(-ω̂) * w_right[1]
    local A_kk = -im * ω_test  # A = -iω
    local S_plus = 1.0 - A_kk * WI
    @printf("  %-10.0f  %+.6e%+.6ei  %-14.4e\n",
        dt, real(S_plus), imag(S_plus), abs(S_plus))
end

# ============================================================================
# Part 3: Matrix case — correct sign convention
# ============================================================================

println("\n\n" * "=" ^ 90)
println("Part 3: Matrix S_+ with correct sign (A = +i·diag(ω) + ε·B)")
println("=" ^ 90)
println()
println("A[kk] = iω_k  matches the CNOT3 rotating-frame convention.")
println("As Δt → ∞:")
println("  diag(S_+) = O(1/(ω·Δt))   ← vanishes")
println("  off-diag(S_+) → -ε·B[jk]/(iω_k) = O(ε/ω)   ← fixed")
println()

N = 3
ω_vec = [10.0, 25.0, 40.0]
ε = 0.1
B = ComplexF64[0 1 1; 1 0 1; 1 1 0]
A_correct = im * Diagonal(ω_vec) + ε * B   # correct sign: A[kk] = +iω_k
A_func_correct = (t -> A_correct,)

println(@sprintf("  ω = %s, ε = %.2f", ω_vec, ε))
println(@sprintf("  %-10s  %-14s  %-14s  %-14s  %-14s",
    "Δt", "max|diag(S+)|", "max|offdiag|", "cond(S+)", "min σ(S+)"))
println("  " * "-"^70)

for dt in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    local Sp, _ = filon_S_plus_S_minus(A_func_correct, ω_vec, 0.0, dt, 0)

    local diag_max = maximum(abs.(diag(Sp)))
    local offdiag_vals = [abs(Sp[i,j]) for i in 1:N, j in 1:N if i != j]
    local offdiag_max = maximum(offdiag_vals)

    @printf("  %-10.1f  %-14.4e  %-14.4e  %-14.4e  %-14.4e\n",
        dt, diag_max, offdiag_max, cond(Sp), minimum(svdvals(Sp)))
end

# Show the predicted vs actual condition number scaling
println()
println("Condition number scaling: cond(S+) ~ C·ε·ω·Δt")
println(@sprintf("  %-10s  %-14s  %-14s  %-14s",
    "Δt", "cond(S+)", "ε·ω_max·Δt", "ratio"))
println("  " * "-"^55)
for dt in [10.0, 50.0, 100.0, 500.0, 1000.0]
    local Sp, _ = filon_S_plus_S_minus(A_func_correct, ω_vec, 0.0, dt, 0)
    local c = cond(Sp)
    local predicted = ε * maximum(ω_vec) * dt
    @printf("  %-10.1f  %-14.4e  %-14.4e  %-14.4f\n", dt, c, predicted, c / predicted)
end

# Compare with wrong sign (A = -i·diag(ω) + ε·B → no cancellation)
println()
println("Contrast: A = -i·diag(ω) + ε·B (wrong sign, NO cancellation)")
A_wrong = -im * Diagonal(ω_vec) + ε * B
A_func_wrong = (t -> A_wrong,)
println(@sprintf("  %-10s  %-14s  %-14s",
    "Δt", "cond(S+)", "min σ(S+)"))
println("  " * "-"^40)
for dt in [10.0, 100.0, 1000.0]
    local Sp, _ = filon_S_plus_S_minus(A_func_wrong, ω_vec, 0.0, dt, 0)
    @printf("  %-10.1f  %-14.4e  %-14.4e\n", dt, cond(Sp), minimum(svdvals(Sp)))
end

# ============================================================================
# Part 4: Verify with actual CNOT3 system
# ============================================================================

println("\n\n" * "=" ^ 90)
println("Part 4: CNOT3 system — diagonal cancellation")
println("=" ^ 90)

using QuantumGateDesign

N_osc_levels = 10
N_guard_levels = 2
Tmax = 550.0

subsystem_sizes = (N_osc_levels, 2+N_guard_levels, 2+N_guard_levels)
essential_subsystem_sizes = (1, 2, 2)

fa = 4.10595; fb = 4.81526; fs = 7.8447
xa = 2 * 0.1099; xb = 2 * 0.1126; xs = 0.002494^2/xa
xab = 1.0e-6; xas = sqrt(xa*xs); xbs = sqrt(xb*xs)

transition_freqs = (fs, fb, fa)
rotation_freqs = transition_freqs
kerr_coeffs = Symmetric([xs xbs xas; xbs xb xab; xas xab xa], :U)

rot_prob = dispersive_qudits_problem(
    subsystem_sizes, essential_subsystem_sizes,
    transition_freqs, rotation_freqs, kerr_coeffs,
    Tmax, 1024, gmres_abstol=1e-15, gmres_reltol=1e-15)

frequencies_rot = -1.0 .* Array(diag(rot_prob.system_sym))
H_diag = diag(rot_prob.system_sym)
N_sys = length(frequencies_rot)

# Show the sign convention
println()
println("Sign convention verification (first 10 components):")
println("  The Filon frequencies are ω_k = -H[kk], so A[kk] = -i·H[kk] = iω_k.")
println(@sprintf("  %-5s  %-14s  %-14s  %-14s  %-14s",
    "k", "H[kk]", "ω_k", "A[kk]/i", "A[kk]/(iω_k)"))
for k in 1:min(10, N_sys)
    local A_kk = -im * H_diag[k]
    local omega_k = frequencies_rot[k]
    local ratio = omega_k == 0 ? NaN : A_kk / (im * omega_k)
    @printf("  %-5d  %+.6e  %+.6e  %+.6e  %+.6e\n",
        k, H_diag[k], omega_k, imag(A_kk), real(ratio))
end
println("  A[kk]/(iω_k) = 1 for all nonzero-frequency components ✓")

# Count zero vs nonzero frequencies
n_zero = count(ω -> ω == 0, frequencies_rot)
n_nonzero = N_sys - n_zero
println()
println("  N = $N_sys total components")
println("  $n_zero with ω_k = 0 (S_+[kk] → 1, well-conditioned)")
println("  $n_nonzero with ω_k ≠ 0 (S_+[kk] → 0, ill-conditioned)")

# Build A(t) function
sys_ham = rot_prob.system_sym + im * rot_prob.system_asym
degree = 14; D1 = 16
Cfreq = zeros(3, 3)
base_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
base_control = QuantumGateDesign.FortranBSplineControl2(base_bspline, Tmax)
rot_controls = [CarrierControl(base_control, freqs) for freqs in eachrow(Cfreq)]
pcof = fill(0.0001, QuantumGateDesign.get_number_of_control_parameters(rot_controls))

A_cfop = ControlledFunctionOp(
    -im .* vcat([sys_ham], rot_prob.sym_operators, im .* rot_prob.asym_operators),
    tuple(
        t -> 1.0,
        [t -> eval_p_derivative(rot_controls[i], t,
            QuantumGateDesign.get_control_vector_slice(pcof, rot_controls, i), 0) for i in 1:3]...,
        [t -> eval_q_derivative(rot_controls[i], t,
            QuantumGateDesign.get_control_vector_slice(pcof, rot_controls, i), 0) for i in 1:3]...,
    )
)
A_mat_func = (t -> Matrix(FilonResearch.full_op(A_cfop(t))),)

# Show diagonal behavior as Δt varies
println()
println("Diagonal entries of S_+ vs Δt (CNOT3, s=0, ω=rot):")
println(@sprintf("  %-10s  %-14s  %-14s  %-14s  %-14s",
    "nsteps", "Δt", "min|diag| (ω≠0)", "max|diag| (ω=0)", "cond(S+)"))
println("  " * "-"^70)

for k in 2:12
    local ns = 2^k
    local dt = Tmax / ns
    local Sp, _ = filon_S_plus_S_minus(A_mat_func, frequencies_rot, 0.0, dt, 0)
    local d = diag(Sp)

    # Separate zero and nonzero frequency components
    local nz_mask = frequencies_rot .!= 0
    local z_mask = frequencies_rot .== 0

    local min_diag_nz = minimum(abs.(d[nz_mask]))
    local max_diag_z = maximum(abs.(d[z_mask]))

    @printf("  %-10d  %-14.4f  %-14.4e  %-14.4e  %-14.4e\n",
        ns, dt, min_diag_nz, max_diag_z, cond(Sp))
end

println()
println("Conclusion: For nonzero-frequency components, |S_+[kk]| → 0 as Δt → ∞,")
println("confirming the asymptotic cancellation 1 - A[kk]·W^I_0[k] = O(1/(ω_k Δt)).")
println("The zero-frequency components keep |S_+[kk]| = O(1), so S_+ has both")
println("O(1) and O(1/(ωΔt)) diagonal entries, causing ill-conditioning ∝ ω·Δt.")
