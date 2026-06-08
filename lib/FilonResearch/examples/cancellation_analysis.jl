"""
Numerical demonstration of floating-point cancellation in the high-frequency
regime for different formulations of the Filon method.

Four approaches for computing D₂u (or equivalently C̃₂f) are compared:

1. "Applied" (matvec): Compute D₂u term-by-term from O(ω²) quantities.
2. "Formed" (backslash): Build D₂ as an explicit matrix, then multiply.
3. "f-form conjugation": Compute Ã = RAR⁻¹ - iΩ via matrix conjugation.
4. "f-form direct": Construct Ã entry-by-entry using frequency differences only.

The test problem is A = iΩ + εB with small off-diagonal coupling ε, so that
A ≈ iΩ and the Filon ansatz is nearly exact. The D_ℓ matrices are O(ε) but
are computed from O(ω^ℓ) terms, creating cancellation.

See Section 4.2 of the Filon notes (main.tex) for the mathematical analysis.
"""

using LinearAlgebra
using Printf

# ============================================================================
# Test problem parameters
# ============================================================================
ε = 0.1
B = ComplexF64[0.0 1.0; 1.0 0.0]  # off-diagonal coupling
u = ComplexF64[1.0, 1.0] / sqrt(2)
t = 0.5  # arbitrary time point for conjugation

ω_values = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]

# ============================================================================
# Part 1: D₁ = A - iΩ
# ============================================================================
println("=" ^ 80)
println("Part 1: Cancellation in D₁u = (A - iΩ)u")
println("=" ^ 80)
println()
println("A = iΩ + εB with ε=$ε. Exact result: D₁u = εBu, |D₁u| = $(round(norm(ε*B*u), sigdigits=4))")
println()
println("When A ≈ iΩ, computing Au - iΩu involves subtracting O(ω) quantities")
println("to obtain an O(ε) result. But forming (A-iΩ) directly avoids this.")
println()
@printf("%-10s  %-14s  %-14s\n", "ω", "Formed", "Applied")
@printf("%-10s  %-14s  %-14s\n", "-"^10, "-"^14, "-"^14)

# True analytical answer: D₁ = εB, so D₁u = εBu
true_D1u = ε * B * u

for ω in ω_values
    freq = [ω, ω + 1.0]
    Omega = Diagonal(freq)
    A = im * Omega + ε * B

    # Method 1: Formed — build (A-iΩ) as a matrix, then multiply
    A_tilde = copy(A)
    for k in 1:2
        A_tilde[k, k] -= im * freq[k]
    end
    result_formed = A_tilde * u

    # Method 2: Applied — compute Au and iΩu separately, subtract
    result_applied = A * u - im .* freq .* u

    err_formed = norm(result_formed - true_D1u)
    err_applied = norm(result_applied - true_D1u)

    @printf("%-10.0e  %-14.3e  %-14.3e\n", ω, err_formed, err_applied)
end

println()
println("Observation: Formed and Applied both give exact results (the iΩ")
println("contribution in A is stored identically, so the subtraction is exact).")

# ============================================================================
# Part 2: D₂ = -Ω² - 2iΩA + Ȧ + A²  (four approaches)
# ============================================================================
println()
println()
println("=" ^ 80)
println("Part 2: Cancellation in D₂u — four approaches")
println("=" ^ 80)
println()
println("Individual terms are O(ω²), but D₂ is O(ε). We compare against the")
println("TRUE analytical answer: D₂ = ε²B² - iε[Ω,B] (since Ȧ=0).")
println()
println("Error hierarchy:")
println("  Applied (matvec):     ~ε_mach·ω²     (catastrophic)")
println("  Formed (backslash):   ~min(ε_mach·ω², ε²)  (saturates at ε²)")
println("  f-form (conjugation): ~ε_mach·ε·ω    (from cis range reduction)")
println("  f-form (direct):      ~ε_mach         (machine precision)")
println()
@printf("%-10s  %-14s  %-14s  %-14s  %-14s  %-10s\n",
    "ω", "Applied", "Formed", "f-conjug", "f-direct", "|D₂u|")
@printf("%-10s  %-14s  %-14s  %-14s  %-14s  %-10s\n",
    "-"^10, "-"^14, "-"^14, "-"^14, "-"^14, "-"^10)

for ω in ω_values
    freq = [ω, ω + 1.0]
    Omega = Diagonal(freq)
    A = im * Omega + ε * B
    dA = zeros(ComplexF64, 2, 2)

    # True analytical D₂: since A = iΩ + εB with Ȧ=0,
    # D₂ = ε²B² - iε[Ω,B], where [Ω,B]_{pk} = (ω_p - ω_k)B_{pk}
    D2_true = ε^2 * B * B
    for p in 1:2, k in 1:2
        D2_true[p, k] -= im * ε * (freq[p] - freq[k]) * B[p, k]
    end
    true_D2u = D2_true * u

    # Method 1: Applied — compute D₂u term-by-term with intermediate matvecs
    Au = A * u
    D2u_applied = -(freq .^ 2) .* u - 2im .* freq .* Au + dA * u + A * Au

    # Method 2: Formed — build D₂ as explicit matrix
    D2_formed = copy(dA) + A * A
    for k in 1:2
        for p in 1:2
            D2_formed[p, k] += -2im * freq[p] * A[p, k]
        end
        D2_formed[k, k] -= freq[k]^2
    end
    result_formed = D2_formed * u

    # Method 3: f-form (conjugation) — compute Ã = RAR⁻¹ - iΩ via cis(ωt)
    # Then C̃₂ = dÃ/dt + Ã² (with dÃ/dt = 0 since dA/dt = 0 and Ã is constant
    # only when A is constant... but A IS constant here, and Ã has time-dependent
    # off-diagonal entries from the conjugation)
    # For constant A: Ã_{pk}(t) = A_{pk} · cis((ω_k - ω_p)t) for p≠k
    #                 Ã_{kk}(t) = A_{kk} - iω_k = εB_{kk}
    # dÃ_{pk}/dt = i(ω_k - ω_p) · Ã_{pk}(t)
    #
    # C̃₂ = dÃ/dt + Ã²  (envelope derivative recursion)
    # But D₂ = R⁻¹ C̃₂ R, and we want D₂u = R⁻¹ C̃₂ f where f = Ru
    #
    # Compute Ã via conjugation (naive: uses cis(ω·t))
    R = Diagonal(cis.(-freq .* t))
    R_inv = Diagonal(cis.(freq .* t))
    A_tilde_conj = R * A * R_inv - im * Omega
    # dÃ/dt via conjugation
    dA_tilde_conj = zeros(ComplexF64, 2, 2)
    for p in 1:2, k in 1:2
        dA_tilde_conj[p, k] = im * (freq[k] - freq[p]) * A_tilde_conj[p, k]
    end
    # C̃₂ = dÃ/dt + Ã²
    C2_tilde_conj = dA_tilde_conj + A_tilde_conj * A_tilde_conj
    # D₂u = R⁻¹ C̃₂ (Ru)
    f = R * u
    D2u_fconj = R_inv * (C2_tilde_conj * f)

    # Method 4: f-form (direct) — construct Ã using frequency DIFFERENCES only
    A_tilde_direct = zeros(ComplexF64, 2, 2)
    for p in 1:2, k in 1:2
        if p == k
            A_tilde_direct[p, k] = A[p, k] - im * freq[k]  # exact subtraction
        else
            Δω = freq[k] - freq[p]
            A_tilde_direct[p, k] = A[p, k] * cis(Δω * t)  # small argument
        end
    end
    # dÃ/dt (direct)
    dA_tilde_direct = zeros(ComplexF64, 2, 2)
    for p in 1:2, k in 1:2
        dA_tilde_direct[p, k] = im * (freq[k] - freq[p]) * A_tilde_direct[p, k]
    end
    # C̃₂ = dÃ/dt + Ã²
    C2_tilde_direct = dA_tilde_direct + A_tilde_direct * A_tilde_direct
    # D₂u = R⁻¹ C̃₂ f, but convert back via frequency differences:
    # [R⁻¹ C̃₂ R]_{pk} = C̃₂_{pk} · cis((ω_p - ω_k)t)
    D2_from_direct = zeros(ComplexF64, 2, 2)
    for p in 1:2, k in 1:2
        Δω = freq[p] - freq[k]
        D2_from_direct[p, k] = C2_tilde_direct[p, k] * cis(Δω * t)
    end
    D2u_fdirect = D2_from_direct * u

    err_applied = norm(D2u_applied - true_D2u)
    err_formed = norm(result_formed - true_D2u)
    err_fconj = norm(D2u_fconj - true_D2u)
    err_fdirect = norm(D2u_fdirect - true_D2u)

    @printf("%-10.0e  %-14.3e  %-14.3e  %-14.3e  %-14.3e  %-10.3e\n",
        ω, err_applied, err_formed, err_fconj, err_fdirect, norm(true_D2u))
end

println()
println("=" ^ 80)
println("Conclusions")
println("=" ^ 80)
println("""
1. For s=0 (2nd order): No cancellation issue in any approach.
   Only D₀ = I appears.

2. For s=1 (4th order): Negligible cancellation in hard-coded approaches.
   D₁ = A - iΩ is computed exactly when A's diagonal entries are stored
   as im*ω_k. The Leibniz-based Algorithm1 shows mild ε_mach·ω growth.

3. For s=2 (6th order): Four distinct error regimes:
   - Applied (matvec): ~ε_mach·ω² — catastrophic, unbounded.
   - Formed (backslash): ~min(ε_mach·ω², ε²) — saturates at ε².
     Loses the ε²B² diagonal contribution when ε² < ε_mach·ω².
   - f-form (conjugation): ~ε_mach·ε·ω — from cis(ωt) range reduction.
   - f-form (direct): ~ε_mach — machine precision at all frequencies.

4. Recommended: Use the direct f-formulation. Construct Ã entry-by-entry
   using frequency differences, compute C̃_ℓ from the envelope recursion,
   then convert back via D_ℓ = R⁻¹ C̃_ℓ R (also using freq differences).
""")
