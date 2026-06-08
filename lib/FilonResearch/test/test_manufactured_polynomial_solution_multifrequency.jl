using FilonResearch, Test, LinearAlgebra, Printf

# ============================================================================
# Manufactured Solution: Single Coupled Chain with Multiple Frequencies
#
# Filon should be much more accurate than Hermite when the frequencies ω_k are
# large, but the differences between successive frequencies Δω are small, since
# then the entries of A(t) can be decently approximated by a polynomial (slow
# oscillation), but u̇(t) cannot be approximated well by a polynomial.
#
# Frequencies are chosen as ω + c_k·Δω so that:
#   - ω is large  →  solution oscillates fast  →  hard for Hermite
#   - Δω is small →  A(t) entries oscillate slowly →  good for Filon
#
# ============================================================================

"""
Return the manufactured solution

    u_k(t) = q_k(t) * exp(i*ω_k*t),   k = 0,1,...,n-1

where
    q_k(t) = sum_{m=0}^k t^m / m!

Examples:
    u_1(t) = exp(i*ω_1*t)
    u_2(t) = (1+t) exp(i*ω_2*t)
    u_3(t) = (1+t+t^2/2) exp(i*ω_3*t)
"""
function mixed_freq_chain_solution(frequencies, t)
    n = length(frequencies)
    @assert n >= 1 "Need at least one frequency"

    sol = zeros(ComplexF64, n)

    q = 1.0
    sol[1] = q * cis(frequencies[1] * t)

    for k in 2:n
        d = k - 1
        q += t^d / factorial(d)
        sol[k] = q * cis(frequencies[k] * t)
    end

    return sol
end

"""
Return A(t) for the mixed-frequency single-chain system

    u'(t) = A(t) u(t),

with
    A[k,k]     = i*ω_k
    A[k,k-1]   = exp(i*(ω_k - ω_{k-1})*t)
"""
function mixed_freq_chain_ode_mat(frequencies, t)
    n = length(frequencies)
    A = zeros(ComplexF64, n, n)

    for k in 1:n
        A[k,k] = im * frequencies[k]
    end

    for k in 2:n
        Δω = frequencies[k] - frequencies[k-1]
        A[k,k-1] = cis(Δω * t)
    end

    return A
end

"""
Return [A(t), A'(t), A''(t), ..., A^(s)(t)] for the mixed-frequency chain.

Diagonal entries are constant, so only the 0th derivative has diagonal terms.
For the subdiagonal:
    d^j/dt^j exp(iΔω t) = (iΔω)^j exp(iΔω t)
"""
function mixed_freq_chain_ode_mat_derivs(frequencies, t, s)
    n = length(frequencies)
    A_derivs = [zeros(ComplexF64, n, n) for _ in 0:s]

    # 0th derivative: A itself
    for k in 1:n
        A_derivs[1][k,k] = im * frequencies[k]
    end

    for j in 0:s
        Aj = A_derivs[j+1]
        for k in 2:n
            Δω = frequencies[k] - frequencies[k-1]
            Aj[k,k-1] = (im * Δω)^j * cis(Δω * t)
        end
    end

    return A_derivs
end

# ============================================================================
# Time-dependent Filon solve wrapper
# ============================================================================

"""
Solve u' = A(t)u on [0,T] with nsteps Filon timesteps, where A is time-dependent.

`A_derivs_func(t, s)` must return
    [A(t), A'(t), ..., A^(s)(t)].
"""
function filon_solve_time_dependent(A_derivs_func, u0, frequencies, T, nsteps, s; rescale=true)
    n = length(u0)
    ts = collect(range(0.0, T, length=nsteps+1))

    U = zeros(ComplexF64, n, nsteps+1)
    U[:,1] = u0

    for step in 1:nsteps
        t_n = ts[step]
        t_np1 = ts[step+1]

        A_derivs_tn   = A_derivs_func(t_n, s)
        A_derivs_tnp1 = A_derivs_func(t_np1, s)

        U[:,step+1] = filon_timestep(
            A_derivs_tn, A_derivs_tnp1,
            U[:,step], frequencies, t_n, t_np1, s;
            rescale=rescale
        )
    end

    return U
end

# ============================================================================
# Printing / convergence helpers
# ============================================================================

function print_triple_table(nsteps_vec, errors_list, orders_list, labels; title="")
    ncols = length(labels)
    println("\n", "="^(12 + 25*ncols))
    println(title)
    println("="^(12 + 25*ncols))
    # Header
    hdr = @sprintf("%10s", "nsteps")
    for label in labels
        hdr *= @sprintf("  │  %15s  %5s", label, "order")
    end
    println(hdr)
    sep = "-"^10
    for _ in labels
        sep *= "──┼──" * "-"^15 * "──" * "-"^5
    end
    println(sep)
    # Rows
    for i in eachindex(nsteps_vec)
        row = @sprintf("%10d", nsteps_vec[i])
        for (errors, orders) in zip(errors_list, orders_list)
            if i == 1
                row *= @sprintf("  │  %15.3e  %5s", errors[i], "-")
            else
                row *= @sprintf("  │  %15.3e  %5.2f", errors[i], orders[i-1])
            end
        end
        println(row)
    end
    # Average orders
    for (label, orders) in zip(labels, orders_list)
        if length(orders) >= 2
            n = min(3, length(orders))
            avg = sum(orders[end-n+1:end]) / n
            println(@sprintf("  %s avg order (last %d): %.2f", label, n, avg))
        end
    end
end

convergence_test_count = Ref(0)

function check_convergence(errors, expected_order; floor=1e-10)
    meaningful = [log2(errors[i] / errors[i+1])
                  for i in 1:length(errors)-1
                  if errors[i] > floor && errors[i+1] > floor]
    if isempty(meaningful)
        @test maximum(errors) < floor
    else
        n = min(3, length(meaningful))
        avg_order = sum(meaningful[end-n+1:end]) / n
        @test avg_order > expected_order - 0.5
    end
    @test errors[end] < errors[1] || errors[end] < floor
    convergence_test_count[] += 1
end

function compute_orders(errors)
    return [log2(errors[i] / errors[i+1])
            for i in 1:length(errors)-1
            if errors[i] > 0 && errors[i+1] > 0]
end

function freq_label(freqs, ω, Δω)
    return @sprintf("ω=%.0f, Δω=%.2f", ω, Δω)
end

# ============================================================================
# Parameters
# ============================================================================

T = 2
nsteps_vec = [2^k for k in 1:8]

# Frequency sets: ω + c_k·Δω with large ω and small Δω
# The coefficients c_k give non-uniform spacing to avoid special structure
freq_coeffs = [0.0, 1.0, 1.3, 2.1, 3.7, 4.2, 5.8]

freq_sets = [
    (ω=10.0,  Δω=0.5),
    (ω=50.0,  Δω=0.5),
    (ω=50.0,  Δω=0.1),
]

# ============================================================================
# Banner
# ============================================================================


println()
println("╔", "═"^58, "╗")
println("║", lpad("Mixed-Frequency Single-Chain Tests", 44), lpad("", 14), "║")
println("║", lpad("", 58), "║")
println("║", lpad("Filon vs Hermite convergence comparison", 50), lpad("", 8), "║")
println("║", lpad("Frequencies: ω + c_k·Δω  (large ω, small Δω)", 51), lpad("", 7), "║")
println("║", lpad("T = $T,  s = 0:3", 28), lpad("", 30), "║")
println("╚", "═"^58, "╝")
println()

# ============================================================================
# Convergence: Filon vs Hermite side by side
# ============================================================================

for (ω, Δω) in freq_sets

    freqs = ω .+ freq_coeffs .* Δω
    u0 = mixed_freq_chain_solution(freqs, 0.0)
    true_sol = mixed_freq_chain_solution(freqs, T)
    Aderivs_func(t, s) = mixed_freq_chain_ode_mat_derivs(freqs, t, s)

    zero_freqs = zeros(length(freqs))
    avg_freq = fill(sum(freqs) / length(freqs), length(freqs))

    println()
    println("━"^78)
    println(@sprintf("  Frequencies: ω = %.1f,  Δω = %.2f", ω, Δω))
    println(@sprintf("  ω_k = [%s]",
        join([@sprintf("%.2f", f) for f in freqs], ", ")))
    println(@sprintf("  avg(ω_k) = %.2f", avg_freq[1]))
    println(@sprintf("  System size: %d,  max polynomial degree: %d", length(freqs), length(freqs)-1))
    println("━"^78)

    for s in 0:3
        # Precompute Filon errors (correct frequencies)
        filon_errors = Float64[]
        for nsteps in nsteps_vec
            sol = filon_solve_time_dependent(
                Aderivs_func, u0, freqs, T, nsteps, s
            )[:, end]
            push!(filon_errors, maximum(abs, sol - true_sol))
        end

        # Precompute Filon errors (average frequency for all components)
        filon_avg_errors = Float64[]
        for nsteps in nsteps_vec
            sol = filon_solve_time_dependent(
                Aderivs_func, u0, avg_freq, T, nsteps, s
            )[:, end]
            push!(filon_avg_errors, maximum(abs, sol - true_sol))
        end

        # Precompute Hermite errors (zero frequencies)
        hermite_errors = Float64[]
        for nsteps in nsteps_vec
            sol = filon_solve_time_dependent(
                Aderivs_func, u0, zero_freqs, T, nsteps, s
            )[:, end]
            push!(hermite_errors, maximum(abs, sol - true_sol))
        end

        filon_orders = compute_orders(filon_errors)
        filon_avg_orders = compute_orders(filon_avg_errors)
        hermite_orders = compute_orders(hermite_errors)

        print_triple_table(
            nsteps_vec,
            [filon_errors, filon_avg_errors, hermite_errors],
            [filon_orders, filon_avg_orders, hermite_orders],
            ["Filon", "Filon(avg ω)", "Hermite"],
            title=@sprintf("s = %d  (expected order: %d)  |  ω = %.1f, Δω = %.2f, T = %g",
                s, 2*(s+1), ω, Δω, T)
        )

        # Check correct order of convergence 
        # (tricky to test only in asymptotic regime)
        # (recommend commenting convergence checks out and inspecting at table output by hand)
        #check_convergence(filon_errors, 2*(s+1))
        #check_convergence(hermite_errors, 2*(s+1))
    end
end

println()
println("="^60)
println("Convergence checks run: $(convergence_test_count[])")
println("="^60)
println("\nDone.")

#end #@testset "Manufactured polynomial/oscillatory solution"
