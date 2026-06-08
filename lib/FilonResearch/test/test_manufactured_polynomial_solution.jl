using FilonResearch, Test, LinearAlgebra, Printf

# ============================================================================
# Manufactured Solution Functions
# ============================================================================

function poly_osc_solution(frequency, degree, t)
    @assert degree >= 0 "Degree must be non-negative"
    sol = zeros(ComplexF64, 1+degree)
    sol[1] = cis(frequency*t)
    for d in 1:degree
        sol[1+d] = sol[d] + ((t^d)/factorial(d))*sol[1]
    end
    return sol
end

function poly_osc_ode_mat(frequency, degree)
    A = zeros(ComplexF64, 1+degree, 1+degree)
    for i in 1:degree
        A[i,i] = im*frequency
        A[i+1,i] = 1
    end
    A[1+degree,1+degree] = im*frequency
    return A
end

function multi_poly_osc_solution(blocks, t)
    n = sum(d + 1 for (_, d) in blocks)
    sol = zeros(ComplexF64, n)

    idx = 1
    for (frequency, degree) in blocks
        @assert degree >= 0
        base = cis(frequency * t)
        sol[idx] = base
        for d in 1:degree
            sol[idx + d] = sol[idx + d - 1] + (t^d / factorial(d)) * base
        end
        idx += degree + 1
    end

    return sol
end

function multi_poly_osc_ode_mat(blocks)
    n = sum(d + 1 for (_, d) in blocks)
    A = zeros(ComplexF64, n, n)

    idx = 1
    for (frequency, degree) in blocks
        m = degree + 1
        for j in 0:degree
            A[idx + j, idx + j] = im * frequency
        end
        for j in 1:degree
            A[idx + j, idx + j - 1] = 1
        end
        idx += m
    end

    return A
end

function multi_poly_osc_frequencies(blocks)
    n = sum(d + 1 for (_, d) in blocks)
    freqs = zeros(Float64, n)

    idx = 1
    for (frequency, degree) in blocks
        for j in 0:degree
            freqs[idx + j] = frequency
        end
        idx += degree + 1
    end

    return freqs
end

function print_convergence_table(nsteps_vec, errors, orders; title="")
    println("\n", "="^60)
    println(title)
    println("="^60)
    println(@sprintf("%10s  %15s  %10s", "nsteps", "error", "order"))
    println("-"^40)
    for i in eachindex(nsteps_vec)
        if i == 1
            println(@sprintf("%10d  %15.3e  %10s", nsteps_vec[i], errors[i], "-"))
        else
            println(@sprintf("%10d  %15.3e  %10.2f", nsteps_vec[i], errors[i], orders[i-1]))
        end
    end
    if length(orders) >= 2
        avg_order = sum(orders[end-min(2,length(orders)-1):end]) / min(3, length(orders))
        println("-"^40)
        println(@sprintf("Average order (last 3): %.2f", avg_order))
    end
end

convergence_test_count = Ref(0)

function check_convergence(errors, expected_order; floor=1e-11)
    meaningful = [log2(errors[i] / errors[i+1])
                  for i in 1:length(errors)-1
                  if errors[i] > floor && errors[i+1] > floor]
    if isempty(meaningful)
        # All errors below floor — method is essentially exact, skip order check
        @test maximum(errors) < floor
    else
        n = min(3, length(meaningful))
        avg_order = sum(meaningful[end-n+1:end]) / n
        @test avg_order > expected_order - 0.5
    end
    @test errors[end] < errors[1] || errors[end] < floor
    convergence_test_count[] += 1
end

# ============================================================================
# Parameters
# ============================================================================

ω = 10.0
T = 1.0

# ============================================================================
# Banner
# ============================================================================

println()
println("╔", "═"^58, "╗")
println("║", lpad("Manufactured Polynomial Solution Tests", 49), lpad("", 9), "║")
println("║", lpad("", 58), "║")
println("║", lpad("Exactness + convergence for p(t)·exp(iωt) solutions", 55), lpad("", 3), "║")
println("║", lpad("ω = $ω,  T = $T,  s = 0:3", 43), lpad("", 15), "║")
println("╚", "═"^58, "╝")
println()

# ============================================================================
# Part 1: Exactness Check (Single Timestep)
# ============================================================================

println("="^60)
println("PART 1: Exactness Check (single timestep, T=$T, ω=$ω)")
println("="^60)
println()
println("Filon with parameter s uses Hermite interpolation of degree 2s+1.")
println("  ω≠0 (Filon): exact for polynomial envelope degree ≤ 2s+1")
println("  ω=0 (Hermite): exact for polynomial degree ≤ 2(s+1)")
println()

for s in 0:3
    max_degree = 2*(s+1) + 1
    println(@sprintf("s = %d:  Filon exact ≤ %d,  Hermite exact ≤ %d", s, 2s+1, 2*(s+1)))
    println(@sprintf("  %6s  %15s  %15s", "degree", "err(ω=$ω)", "err(ω=0)"))
    println("  ", "-"^40)

    for degree in 0:max_degree
        N = degree + 1

        # Filon (exact frequencies)
        A = poly_osc_ode_mat(ω, degree)
        u0 = ones(ComplexF64, N)
        true_sol = poly_osc_solution(ω, degree, T)
        sol_filon = filon_timestep(A, u0, fill(ω, N), 0.0, T, s)
        err_filon = maximum(abs, sol_filon - true_sol)

        # Hermite (ω=0 system, zero frequencies)
        A0 = poly_osc_ode_mat(0.0, degree)
        true_sol0 = poly_osc_solution(0.0, degree, T)
        sol_hermite = filon_timestep(A0, real.(u0), zeros(N), 0.0, T, s)
        err_hermite = maximum(abs, sol_hermite - true_sol0)

        println(@sprintf("  %6d  %15.3e  %15.3e", degree, err_filon, err_hermite))

        # Filon exactness: degree ≤ 2s+1
        if degree <= 2s+1
            @test err_filon < 1e-13
        end
        # Hermite exactness: degree ≤ 2(s+1)
        if degree <= 2*(s+1)
            @test err_hermite < 1e-13
        end
    end
    println()
end

# ============================================================================
# Part 2: Convergence with Exact Filon Frequencies
# ============================================================================

println()
println("="^60)
println("PART 2: Convergence (exact Filon frequencies, ω=$ω)")
println("="^60)
println("Solution envelope degree 10, exceeds 2s+1 for all s ≤ 3.")
println("Expected order: 2(s+1)")

degree = 10
N = degree + 1
A = poly_osc_ode_mat(ω, degree)
u0 = ones(ComplexF64, N)
frequencies = fill(ω, N)
true_sol = poly_osc_solution(ω, degree, T)
nsteps_vec = [2^k for k in 1:10]

for s in 0:3
    errors = Float64[]
    for nsteps in nsteps_vec
        sol = filon_solve(A, u0, frequencies, T, nsteps, s)[:, end]
        push!(errors, maximum(abs, sol - true_sol))
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1 if errors[i] > 0 && errors[i+1] > 0]
    print_convergence_table(nsteps_vec, errors, orders,
        title="s = $s (expected order: $(2*(s+1)))")

    check_convergence(errors, 2*(s+1))
end

# ============================================================================
# Part 3: Convergence with Zero Frequencies (Hermite on oscillatory solution)
# ============================================================================

println()
println("="^60)
println("PART 3: Convergence (zero frequencies / Hermite, ω=$ω)")
println("="^60)
println("Same oscillatory solution as Part 2, but frequencies = zeros.")
println("Hermite treats p(t)·e^(iωt) as a generic function — never exact.")
println("Expected order: 2(s+1)")

zero_freqs = zeros(N)

for s in 0:3
    errors = Float64[]
    for nsteps in nsteps_vec
        sol = filon_solve(A, u0, zero_freqs, T, nsteps, s)[:, end]
        push!(errors, maximum(abs, sol - true_sol))
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1 if errors[i] > 0 && errors[i+1] > 0]
    print_convergence_table(nsteps_vec, errors, orders,
        title="s = $s (expected order: $(2*(s+1)))")

    check_convergence(errors, 2*(s+1))
end

# ============================================================================
# Part 4: Multi-Frequency Exactness Check
# ============================================================================

println()
println("="^60)
println("PART 4: Multi-Frequency Exactness Check (single timestep, T=$T)")
println("="^60)
println()
println("Block-diagonal Jordan system with multiple frequencies.")
println("Each block has its own frequency and polynomial degree.")
println("Filon with exact frequencies should be exact when all block degrees ≤ 2s+1.")
println()

blocks_list = [
    [(ω, 0), (2ω, 0)],
    [(ω, 1), (2ω, 1)],
    [(ω, 1), (2ω, 2)],
    [(ω, 2), (2ω, 2), (3ω, 1)],
]

for s in 0:3
    println(@sprintf("s = %d (Filon exact for envelope degree ≤ %d):", s, 2s+1))
    println(@sprintf("  %-40s  %15s", "blocks", "error"))
    println("  ", "-"^58)

    for blocks in blocks_list
        local N = sum(d + 1 for (_, d) in blocks)
        local A = multi_poly_osc_ode_mat(blocks)
        local u0 = multi_poly_osc_solution(blocks, 0.0)
        local true_sol = multi_poly_osc_solution(blocks, T)
        freqs = multi_poly_osc_frequencies(blocks)

        sol = filon_timestep(A, u0, freqs, 0.0, T, s)
        err = maximum(abs, sol - true_sol)

        max_deg = maximum(d for (_, d) in blocks)
        label = join(["(ω=$(Int(f/ω))ω, d=$d)" for (f, d) in blocks], ", ")
        println(@sprintf("  %-40s  %15.3e", label, err))

        # Exact when all block degrees ≤ 2s+1
        if max_deg <= 2s+1
            @test err < 1e-12
        end
    end
    println()
end

# ============================================================================
# Part 5: Multi-Frequency Convergence with Exact Filon Frequencies
# ============================================================================

println()
println("="^60)
println("PART 5: Multi-Frequency Convergence (exact frequencies)")
println("="^60)
println("Blocks with high-degree envelopes exceeding 2s+1 for all s ≤ 3.")
println("Expected order: 2(s+1)")

conv_blocks = [(ω, 5), (2ω, 6)]
N_multi = sum(d + 1 for (_, d) in conv_blocks)
A_multi = multi_poly_osc_ode_mat(conv_blocks)
u0_multi = multi_poly_osc_solution(conv_blocks, 0.0)
freqs_multi = multi_poly_osc_frequencies(conv_blocks)
true_sol_multi = multi_poly_osc_solution(conv_blocks, T)

for s in 0:3
    errors = Float64[]
    for nsteps in nsteps_vec
        sol = filon_solve(A_multi, u0_multi, freqs_multi, T, nsteps, s)[:, end]
        push!(errors, maximum(abs, sol - true_sol_multi))
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1 if errors[i] > 0 && errors[i+1] > 0]
    print_convergence_table(nsteps_vec, errors, orders,
        title="s = $s (expected order: $(2*(s+1)))")

    check_convergence(errors, 2*(s+1))
end

# ============================================================================
# Part 6: Multi-Frequency Convergence with Zero Frequencies (Hermite)
# ============================================================================

println()
println("="^60)
println("PART 6: Multi-Frequency Convergence (zero frequencies / Hermite)")
println("="^60)
println("Same multi-frequency solution, but Filon frequencies all set to zero.")
println("Expected order: 2(s+1)")

zero_freqs_multi = zeros(N_multi)

for s in 0:3
    errors = Float64[]
    for nsteps in nsteps_vec
        sol = filon_solve(A_multi, u0_multi, zero_freqs_multi, T, nsteps, s)[:, end]
        push!(errors, maximum(abs, sol - true_sol_multi))
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1 if errors[i] > 0 && errors[i+1] > 0]
    print_convergence_table(nsteps_vec, errors, orders,
        title="s = $s (expected order: $(2*(s+1)))")

    check_convergence(errors, 2*(s+1))
end

println()
println("="^60)
println("Convergence checks run: $(convergence_test_count[])")
println("="^60)
println("\nDone.")
