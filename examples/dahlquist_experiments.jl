"""
Numerical experiments on the Dahlquist equation:

    du/dt = λu,  u(0) = u₀

with the ansatz u(t) = f(t)e^(iωt).

Three cases of interest:
1. On resonance: λ = iω
2. Off resonance: λ = iω̃ ≠ iω, |ω - ω̃| ≪ 1
3. Off resonance and damped: λ = iω̃ - κ, |κ/ω̃| ≪ 1, κ > 0
"""

using FilonResearch
using Plots
using Printf

# Exact solution for the Dahlquist equation
dahlquist_exact(λ, u₀, t) = u₀ * exp(λ * t)

"""
Solve the Dahlquist equation using the Filon method with nsteps timesteps.
Returns the solution at time tf.
"""
function filon_dahlquist(λ, ω, s, nsteps, tf, u₀; rescale=true)
    dt = tf / nsteps
    u = ComplexF64(u₀)
    for n in 1:nsteps
        t_n = (n - 1) * dt
        t_np1 = n * dt
        u = filon_timestep(λ, ω, u, s, t_n, t_np1; rescale=rescale)
    end
    return u
end

"""
Run a convergence study: compute errors for a range of nsteps values.
Returns (nsteps_vec, errors).
"""
function convergence_study(λ, ω, s, u₀, tf, nsteps_range; rescale=true)
    u_exact = dahlquist_exact(λ, u₀, tf)
    errors = Float64[]
    for nsteps in nsteps_range
        u_numerical = filon_dahlquist(λ, ω, s, nsteps, tf, u₀; rescale=rescale)
        push!(errors, abs(u_numerical - u_exact))
    end
    return collect(nsteps_range), errors
end

# =============================================================================
# Main experiment parameters
# =============================================================================

ω = 10.0                    # Ansatz frequency
tf = 10.0                   # Final time
u₀ = 1.0 + 0.0im            # Initial condition
nsteps_range = [2^k for k in 1:12]  # Number of timesteps
s_values = 0:2              # Orders to test

# Consistent styling: color distinguishes s
s_colors = [:blue, :red, :green]

println("=" ^ 70)
println("Dahlquist Equation Experiments")
println("=" ^ 70)
println("Ansatz frequency ω = $ω")
println("Final time tf = $tf")
println("Initial condition u₀ = $u₀")
println()

# =============================================================================
# Case 1: On resonance (λ = iω)
# =============================================================================

println("-" ^ 70)
println("Case 1: On resonance (λ = iω)")
println("-" ^ 70)

λ_resonance = im * ω

println("λ = $(λ_resonance)")
println("Expected: Exact solution for all s (method is exact when λ = iω)")
println()

p1 = plot(title="Case 1: On resonance (λ = iω)",
          xlabel="Number of timesteps",
          ylabel="Error |u_numerical - u_exact|",
          xscale=:log10, yscale=:log10,
          legend=:topright)

for (i, s) in enumerate(s_values)
    nsteps_vec, errors = convergence_study(λ_resonance, ω, s, u₀, tf, nsteps_range)
    # Handle zero/near-zero errors for plotting
    errors_plot = max.(errors, 1e-16)
    plot!(p1, nsteps_vec, errors_plot, marker=:circle, color=s_colors[i],
          linestyle=:solid, label="s=$s")
    @printf("  s=%d: max error = %.2e, min error = %.2e\n", s, maximum(errors), minimum(errors))
end
println()

# =============================================================================
# Case 2: Off resonance (λ = iω̃, |ω - ω̃| ≪ 1)
# =============================================================================

println("-" ^ 70)
println("Case 2: Off resonance (λ = iω̃, |ω - ω̃| ≪ 1)")
println("-" ^ 70)

detunings = [0.1, 0.01, 0.001]
Δω_markers = [:circle, :square, :diamond]

p2 = plot(title="Case 2: Off resonance (λ = iω̃)",
          xlabel="Number of timesteps",
          ylabel="Error |u_numerical - u_exact|",
          xscale=:log10, yscale=:log10,
          legend=:outerright)

for (j, Δω) in enumerate(detunings)
    ω̃ = ω + Δω
    λ_offresonance = im * ω̃
    println("Detuning Δω = $Δω (ω̃ = $ω̃, λ = $(λ_offresonance))")

    for (i, s) in enumerate(s_values)
        nsteps_vec, errors = convergence_study(λ_offresonance, ω, s, u₀, tf, nsteps_range)
        @printf("  s=%d: max error = %.2e, min error = %.2e\n",
                s, maximum(errors), minimum(errors))
        plot!(p2, nsteps_vec, errors, marker=Δω_markers[j], color=s_colors[i],
              linestyle=:solid, label="Δω=$Δω, s=$s")
    end
    println()
end

# Reference lines for convergence orders (shifted down)
plot!(p2, nsteps_range, 1e1 ./ (nsteps_range.^2),
      label="O(n⁻²)", linestyle=:dot, color=:gray, linewidth=2)
plot!(p2, nsteps_range, 1e-1 ./ (nsteps_range.^4),
      label="O(n⁻⁴)", linestyle=:dash, color=:gray, linewidth=2)
plot!(p2, nsteps_range, 1e-3 ./ (nsteps_range.^6),
      label="O(n⁻⁶)", linestyle=:dashdot, color=:gray, linewidth=2)

# =============================================================================
# Case 3: Off resonance and damped (λ = iω̃ - κ, κ > 0)
# =============================================================================

println("-" ^ 70)
println("Case 3: Off resonance and damped (λ = iω̃ - κ)")
println("-" ^ 70)

# Test with small damping: |κ/ω̃| ≪ 1
κ = 0.1

p3 = plot(title="Case 3: Off resonance and damped (κ=$κ)",
          xlabel="Number of timesteps",
          ylabel="Error |u_numerical - u_exact|",
          xscale=:log10, yscale=:log10,
          legend=:outerright)

for (j, Δω) in enumerate(detunings)
    ω̃ = ω + Δω
    λ_damped = im * ω̃ - κ
    println("κ = $κ, Δω = $Δω (ω̃ = $ω̃, λ = $(λ_damped), κ/ω̃ = $(κ/ω̃))")

    for (i, s) in enumerate(s_values)
        nsteps_vec, errors = convergence_study(λ_damped, ω, s, u₀, tf, nsteps_range)
        @printf("  s=%d: max error = %.2e, min error = %.2e\n",
                s, maximum(errors), minimum(errors))
        plot!(p3, nsteps_vec, errors, marker=Δω_markers[j], color=s_colors[i],
              linestyle=:solid, label="Δω=$Δω, s=$s")
    end
    println()
end

# Reference lines (shifted down)
plot!(p3, nsteps_range, 1e1 ./ (nsteps_range.^2),
      label="O(n⁻²)", linestyle=:dot, color=:gray, linewidth=2)
plot!(p3, nsteps_range, 1e-1 ./ (nsteps_range.^4),
      label="O(n⁻⁴)", linestyle=:dash, color=:gray, linewidth=2)
plot!(p3, nsteps_range, 1e-3 ./ (nsteps_range.^6),
      label="O(n⁻⁶)", linestyle=:dashdot, color=:gray, linewidth=2)

# =============================================================================
# Summary plot
# =============================================================================

println("=" ^ 70)
println("Generating summary plot...")
println("=" ^ 70)

summary_plot = plot(p1, p2, p3, layout=@layout([a b; c]), size=(1200, 800))

savefig(summary_plot, joinpath(@__DIR__, "..", "Plots", "dahlquist_experiments.png"))
println("Saved figure to Plots/dahlquist_experiments.png")

# Display individual plots if running interactively
display(summary_plot)
