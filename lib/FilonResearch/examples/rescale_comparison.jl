using FilonResearch
using LinearAlgebra
using Printf

println("""
This script compares the accuracy of the taking a single timestep using the Filon
method with and without rescaling the timestep to the interval [-1,1].

The condition number of the confluent Vandermonde matrix which is used as the
left-hand side when solving for monomial coefficients when computing the
weights scales (I think) with the distance between nodes, the absolute values
of the nodes, and the size of the matrix (which is determined here by s, the
number of derivatives to take).

Therefore, as the timestep size decreases, the starting time increases, or s
increases, we expect the rescaled version to be more accurate than the unrescaled
version, with the latter eventually breaking down due to the confluent
Vandermonde matrix becoming singular.
""")

# Helper function to run a single comparison test
# Returns (err_rescaled, err_not_rescaled, singular_error)
# where singular_error is true if the non-rescaled version threw a singular matrix error
function run_comparison(s, ws, t_n, dt)
    A = Diagonal(ws .* im)
    u_n = ones(ComplexF64, length(ws))
    A_derivs = [i == 0 ? A : zero(A) for i in 0:s]

    u_rescaled = filon_timestep(A_derivs, A_derivs, u_n, ws, t_n, dt, s; rescale=true)
    u_true = u_n .* cis.(ws .* dt)
    err_rescaled = maximum(abs.(u_rescaled - u_true))

    # Try the non-rescaled version, catching singular matrix errors
    err_not_rescaled = NaN
    singular_error = false
    try
        u_not_rescaled = filon_timestep(A_derivs, A_derivs, u_n, ws, t_n, dt, s; rescale=false)
        err_not_rescaled = maximum(abs.(u_not_rescaled - u_true))
    catch e
        if e isa SingularException
            singular_error = true
        else
            rethrow(e)
        end
    end

    return err_rescaled, err_not_rescaled, singular_error
end

# Helper function to format the error, showing "SINGULAR" if a singular error occurred
function format_error(err, singular)
    if singular
        return "SINGULAR"
    else
        return @sprintf("%-14.2e", err)
    end
end

# =============================================================================
# Effect 1: Decreasing timestep size (distance between nodes)
# =============================================================================
println("=" ^ 70)
println("Effect 1: Decreasing timestep size (distance between nodes)")
println("=" ^ 70)
println("As dt → 0, the nodes t_n and t_n + dt become closer together,")
println("increasing the condition number of the confluent Vandermonde matrix.")
println()

ws = [1.0]
t_n = 0.0
timesteps = [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

for s in 0:3
    println("s = $s")
    @printf("  %-12s  %-12s  %-14s  %-14s\n", "t_n", "dt", "Err (rescaled)", "Err (not rescaled)")
    @printf("  %-12s  %-12s  %-14s  %-14s\n", "-" ^ 12, "-" ^ 12, "-" ^ 14, "-" ^ 14)

    for dt in timesteps
        err_r, err_nr, singular = run_comparison(s, ws, t_n, dt)
        @printf("  %-12.0e  %-12.0e  %-14.2e  %s\n", t_n, dt, err_r, format_error(err_nr, singular))
    end
    println()
end

# =============================================================================
# Effect 2: Increasing starting time (absolute values of nodes)
# =============================================================================
println("=" ^ 70)
println("Effect 2: Increasing starting time (absolute values of nodes)")
println("=" ^ 70)
println("As t_n increases, the absolute values of both nodes grow,")
println("increasing the condition number of the confluent Vandermonde matrix.")
println()

ws = [1.0]
dt = 0.01
start_times = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]

for s in 0:3
    println("s = $s")
    @printf("  %-12s  %-12s  %-14s  %-14s\n", "t_n", "dt", "Err (rescaled)", "Err (not rescaled)")
    @printf("  %-12s  %-12s  %-14s  %-14s\n", "-" ^ 12, "-" ^ 12, "-" ^ 14, "-" ^ 14)

    for t_n in start_times
        err_r, err_nr, singular = run_comparison(s, ws, t_n, dt)
        @printf("  %-12.0e  %-12.0e  %-14.2e  %s\n", t_n, dt, err_r, format_error(err_nr, singular))
    end
    println()
end

# =============================================================================
# Effect 3: Increasing s (matrix size)
# =============================================================================
println()
println("=" ^ 70)
println("Effect 3: Increasing s (number of derivatives)")
println("=" ^ 70)
println("As s increases, the confluent Vandermonde matrix grows in size,")
println("increasing its condition number.")
println()

ws = [1.0]
t_n = 0.0
dt = 0.1
s_values = [0, 1, 2, 3]

@printf("  %-4s  %-12s  %-12s  %-14s  %-14s\n", "s", "t_n", "dt", "Err (rescaled)", "Err (not rescaled)")
@printf("  %-4s  %-12s  %-12s  %-14s  %-14s\n", "-" ^ 4, "-" ^ 12, "-" ^ 12, "-" ^ 14, "-" ^ 14)

for s in s_values
    err_r, err_nr, singular = run_comparison(s, ws, t_n, dt)
    @printf("  %-4d  %-12.0e  %-12.0e  %-14.2e  %s\n", s, t_n, dt, err_r, format_error(err_nr, singular))
end

# =============================================================================
# Combined effects: small dt + large t_n
# =============================================================================
println("=" ^ 70)
println("Combined effects: small dt + large t_n")
println("=" ^ 70)
println("When multiple factors combine, the non-rescaled version breaks down")
println("more dramatically.")
println()

ws = [1.0]
cases = [
    (0.0, 1.0),
    (0.0, 0.01),
    (100.0, 0.01),
    (1e4, 0.001),
    (1e6, 0.0001),
]

for s in 0:3
    println("s = $s")
    @printf("  %-12s  %-12s  %-14s  %-14s\n", "t_n", "dt", "Err (rescaled)", "Err (not rescaled)")
    @printf("  %-12s  %-12s  %-14s  %-14s\n", "-" ^ 12, "-" ^ 12, "-" ^ 14, "-" ^ 14)

    for (t_n, dt) in cases
        err_r, err_nr, singular = run_comparison(s, ws, t_n, dt)
        @printf("  %-12.0e  %-12.0e  %-14.2e  %s\n", t_n, dt, err_r, format_error(err_nr, singular))
    end
    println()
end

println()
println("=" ^ 70)
println("Done.")
