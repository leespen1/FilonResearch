#=
3×3 Oscillatory ODE Convergence Test
=====================================

Verifies correctness of the Filon method on a 3×3 matrix ODE system:
    du/dt = A(t)u,  where A(t) = i*D + A₀(t)

D = diag(ω₁, ω₂, ω₃) with large distinct entries, A₀(t) a small smooth perturbation.

We compare:
1. Filon with correct frequencies (ω_k = D_kk) — should converge rapidly
2. Filon with half frequencies (ω_k/2) — intermediate case
3. Filon with frequencies = 0 (pure Hermite) — standard 2s+2 convergence but needs many more steps
=#

using FilonResearch
using LinearAlgebra
using Printf

# ============================================================================
# Helpers (reused from convergence_order_test.jl)
# ============================================================================

function compute_convergence_order(errors, refinement_factor=2)
    orders = Float64[]
    for i in 1:length(errors)-1
        if errors[i] > 0 && errors[i+1] > 0
            order = log(errors[i] / errors[i+1]) / log(refinement_factor)
            push!(orders, order)
        end
    end
    return orders
end

"""
Print a side-by-side convergence table comparing multiple methods.
`cases` is a vector of (label, errors) pairs.
"""
function print_comparison_table(nsteps_vec, cases; title="")
    ncases = length(cases)
    # Build header
    w = 10 + ncases * 22 + (ncases - 1) * 5
    println("\n", "="^w)
    println(title)
    println("="^w)
    header = @sprintf("%10s", "nsteps")
    for (label, _) in cases
        header *= @sprintf("  │  %12s  %6s", "err($label)", "order")
    end
    println(header)
    println("-"^w)

    # Compute orders for each case
    all_orders = [compute_convergence_order(errs) for (_, errs) in cases]

    for i in eachindex(nsteps_vec)
        line = @sprintf("%10d", nsteps_vec[i])
        for (j, (_, errs)) in enumerate(cases)
            ord = i == 1 ? "  -   " : @sprintf("%6.2f", all_orders[j][i-1])
            line *= @sprintf("  │  %12.3e  %s", errs[i], ord)
        end
        println(line)
    end
    println("-"^w)

    # Average orders
    avg_line = @sprintf("%10s", "avg(last3)")
    for (j, _) in enumerate(cases)
        ords = all_orders[j]
        if length(ords) >= 2
            avg = sum(ords[end-min(2,length(ords)-1):end]) / min(3, length(ords))
            avg_line *= @sprintf("  │  %12s  %6.2f", "", avg)
        else
            avg_line *= @sprintf("  │  %12s  %6s", "", "-")
        end
    end
    println(avg_line)
end

# ============================================================================
# Problem Setup
# ============================================================================

# Diagonal frequencies (large, distinct)
ω₁, ω₂, ω₃ = 10.0, 25.0, 40.0
#ω₁, ω₂, ω₃ = 0.0, 0.0, 0.0
D = Diagonal([ω₁, ω₂, ω₃])

# Small smooth perturbation and its derivatives
#ε = 0.1
ε = 0.0

A₀(t) = ComplexF64.(ε * [sin(t)    cos(t)    sin(2t);
                         cos(t)    sin(3t)   cos(t);
                         sin(2t)   cos(t)    sin(t)])

dA₀(t) = ComplexF64.(ε * [cos(t)     -sin(t)    2cos(2t);
                           -sin(t)    3cos(3t)   -sin(t);
                           2cos(2t)   -sin(t)    cos(t)])

d2A₀(t) = ComplexF64.(ε * [-sin(t)    -cos(t)    -4sin(2t);
                            -cos(t)    -9sin(3t)  -cos(t);
                            -4sin(2t)  -cos(t)    -sin(t)])

d3A₀(t) = ComplexF64.(ε * [-cos(t)     sin(t)    -8cos(2t);
                            sin(t)     -27cos(3t)  sin(t);
                            -8cos(2t)   sin(t)    -cos(t)])

# Full A(t) = im*D + A₀(t), derivatives of im*D are zero
imD = ComplexF64.(im * D)
A_deriv_funcs = (
    t -> imD + A₀(t),
    t -> dA₀(t),
    t -> d2A₀(t),
    t -> d3A₀(t),
)

# Initial condition and final time
u₀ = ComplexF64[1.0, 1.0, 1.0]
T = 200.0

# Frequencies for the Filon ansatz
frequencies_filon = [ω₁, ω₂, ω₃]
frequencies_half = [ω₁, ω₂, ω₃] ./ 2
frequencies_zero = [0.0, 0.0, 0.0]

# ============================================================================
# Reference Solution
# ============================================================================

println("Computing reference solution with 2^14 steps...")
nsteps_ref = 2^14
u_ref = filon_solve(A_deriv_funcs, u₀, frequencies_filon, T, nsteps_ref, 3, rescale=true)
u_exact = u_ref[:, end]
println(@sprintf("Reference solution: norm = %.10e", norm(u_exact)))

# ============================================================================
# Convergence Sweep
# ============================================================================

n_refinements = 10
nsteps_vec = [2^k for k in 0:n_refinements]
s_values = [0, 1, 2, 3]

for s in s_values
    if s >= length(A_deriv_funcs)
        println("Skipping s=$s: not enough derivatives provided")
        continue
    end

    errors_filon = Float64[]
    errors_half = Float64[]
    errors_hermite = Float64[]
    for nsteps in nsteps_vec
        sol_f = filon_solve(A_deriv_funcs, u₀, frequencies_filon, T, nsteps, s, rescale=true)
        push!(errors_filon, norm(sol_f[:, end] - u_exact))

        sol_half = filon_solve(A_deriv_funcs, u₀, frequencies_half, T, nsteps, s, rescale=true)
        push!(errors_half, norm(sol_half[:, end] - u_exact))

        sol_h = filon_solve(A_deriv_funcs, u₀, frequencies_zero, T, nsteps, s, rescale=true)
        push!(errors_hermite, norm(sol_h[:, end] - u_exact))
    end

    expected_order = 2s + 2
    print_comparison_table(nsteps_vec,
        [("ω", errors_filon), ("ω/2", errors_half), ("ω=0", errors_hermite)],
        title="s = $s (expected order: $expected_order)")
end

println("\n" * "="^70)
println("Done.")
println("="^70)
