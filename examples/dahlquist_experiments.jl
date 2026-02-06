"""
Numerical experiments on the Dahlquist equation:

    du/dt = λu,  u(0) = u₀

with the ansatz u(t) = f(t)e^(iωt).

λ is fixed (λ = 10i) and ω (the ansatz frequency) is varied:
1. On resonance: ω = Im(λ)
2. Off resonance: ω = Im(λ) + Δω
3. Off resonance and damped: λ → λ - κ, ω = Im(λ) + Δω
"""

using FilonResearch
using CairoMakie
using LaTeXStrings
using Printf

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

# Exact solution for the Dahlquist equation
dahlquist_exact(λ, u₀, t) = u₀ * exp(λ * t)

"""
Solve the Dahlquist equation using the Filon method with nsteps timesteps.
Returns the full trajectory as (times, solutions).
"""
function filon_dahlquist(λ, ω, s, nsteps, tf, u₀; rescale=true)
    dt = tf / nsteps
    u = ComplexF64(u₀)
    times = Vector{Float64}(undef, nsteps + 1)
    sols = Vector{ComplexF64}(undef, nsteps + 1)
    times[1] = 0.0
    sols[1] = u
    for n in 1:nsteps
        t_n = (n - 1) * dt
        t_np1 = n * dt
        u = filon_timestep(λ, ω, u, s, t_n, t_np1; rescale=rescale)
        times[n + 1] = t_np1
        sols[n + 1] = u
    end
    return times, sols
end

"""
Compute the discrete L2 norm of the error over [0, tf].
Uses ||e||_h = √(h ∑ᵢ |eᵢ|²) where h = Δt is the grid spacing.
"""
function l2_error(times, sols, λ, u₀)
    exact = [dahlquist_exact(λ, u₀, t) for t in times]
    h = times[2] - times[1]
    return sqrt(h * sum(abs2, sols .- exact))
end

"""
Run a convergence study: compute L2 errors for a range of nsteps values.
Returns (nsteps_vec, errors).
"""
function convergence_study(λ, ω, s, u₀, tf, nsteps_range; rescale=true)
    errors = Float64[]
    for nsteps in nsteps_range
        times, sols = filon_dahlquist(λ, ω, s, nsteps, tf, u₀; rescale=rescale)
        push!(errors, l2_error(times, sols, λ, u₀))
    end
    return collect(nsteps_range), errors
end

# =============================================================================
# Main experiment parameters
# =============================================================================

λ = 10.0im                  # System eigenvalue (fixed)
tf = 2.5pi                     # Final time
u₀ = 1.0 + 0.0im            # Initial condition
nsteps_range = [2^k for k in 1:12]  # Number of timesteps
s_values = 0:3              # Orders to test

colors = Makie.wong_colors()

println("=" ^ 70)
println("Dahlquist Equation Experiments")
println("=" ^ 70)
println("System eigenvalue λ = $λ")
println("Final time tf = $tf")
println("Initial condition u₀ = $u₀")
println()

inch = 96  # points per inch
fig = Figure(size=(6.5inch, 6.5inch), fontsize=12)

# Main title
tf_over_pi = tf / π
tf_str = isinteger(tf_over_pi) ? "$(Int(tf_over_pi))\\pi" : "$(tf_over_pi)\\pi"
Label(fig[0, 1:3], L"\textbf{Filon Method on } du/dt = \lambda u \textbf{ with ansatz } u(t) = f(t)e^{i\omega t}, \quad 0 \leq t \leq %$(tf_str)",
    fontsize=16, padding=(0, 0, 0, 0))
#rowgap!(fig.layout, 1, 5)

# =============================================================================
# Case 1: On resonance (ω = Im(λ))
# =============================================================================

println("-" ^ 70)
println("On resonance (ω = Im(λ))")
println("-" ^ 70)

ω_resonance = imag(λ)

println("λ = $λ, ω = $ω_resonance")
println("Expected: Exact solution for all s (method is exact when ω = Im(λ))")
println()

ax1 = Axis(fig[1, 1],
    title=L"\lambda = %$(Int(imag(λ)))i, \quad \omega = %$(Int(imag(λ)))",
    titlegap=10,
    xlabel="Number of Timesteps",
    ylabel=L"L^2 \textrm{ Error}",
    xscale=log10, yscale=log10)

for (i, s) in enumerate(s_values)
    nsteps_vec, errors = convergence_study(λ, ω_resonance, s, u₀, tf, nsteps_range)
    errors_plot = max.(errors, 1e-16)
    scatterlines!(ax1, nsteps_vec, errors_plot, color=colors[i],
        marker=:circle)
    @printf("  s=%d: max error = %.2e, min error = %.2e\n", s, maximum(errors), minimum(errors))
end
println()

# =============================================================================
# Case 2: Off resonance (ω = Im(λ) + Δω)
# =============================================================================

println("-" ^ 70)
println("Off resonance (ω = Im(λ) + Δω)")
println("-" ^ 70)

detunings = [0.1, 0.01, 0.001]
detunings_damped = [0.1, 0.01, 0.001]
markers = [:circle, :rect, :diamond]
markers_damped = [:circle, :rect, :diamond]

ax2 = Axis(fig[1, 2],
    title=L"\lambda = %$(Int(imag(λ)))i, \quad \omega = %$(Int(imag(λ))) + \Delta\omega",
    titlegap=10,
    xlabel="Number of Timesteps",
    yticklabelsvisible=false,
    xscale=log10, yscale=log10)

for (j, Δω) in enumerate(detunings)
    ω_off = imag(λ) + Δω
    println("Detuning Δω = $Δω (ω = $ω_off)")

    for (i, s) in enumerate(s_values)
        nsteps_vec, errors = convergence_study(λ, ω_off, s, u₀, tf, nsteps_range)
        @printf("  s=%d: max error = %.2e, min error = %.2e\n",
                s, maximum(errors), minimum(errors))
        scatterlines!(ax2, nsteps_vec, errors, color=colors[i],
            marker=markers[j])
    end
    println()
end

# Reference lines (all dotted grey)
lines!(ax2, nsteps_range, 1e1 ./ (nsteps_range .^ 2),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax2, nsteps_range, 1e-1 ./ (nsteps_range .^ 4),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax2, nsteps_range, 1e-3 ./ (nsteps_range .^ 6),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax2, nsteps_range, 1e-5 ./ (nsteps_range .^ 8),
    linestyle=:dot, color=:gray, linewidth=2)

# =============================================================================
# Case 3: Off resonance and damped (λ → λ - κ, ω = Im(λ) + Δω)
# =============================================================================

println("-" ^ 70)
println("Off resonance and damped (λ → λ - κ)")
println("-" ^ 70)

κ = 0.01
λ_damped = λ - κ

ax3 = Axis(fig[2, 1:2],
    title=L"\lambda = %$(Int(imag(λ)))i - %$(κ), \quad \omega = %$(Int(imag(λ))) + \Delta\omega",
    titlegap=10,
    xlabel="Number of Timesteps",
    ylabel=L"L^2 \textrm{ Error}",
    xscale=log10, yscale=log10)

for (j, Δω) in enumerate(detunings_damped)
    ω_off = imag(λ) + Δω
    println("κ = $κ, Δω = $Δω (λ = $λ_damped, ω = $ω_off)")

    for (i, s) in enumerate(s_values)
        nsteps_vec, errors = convergence_study(λ_damped, ω_off, s, u₀, tf, nsteps_range)
        @printf("  s=%d: max error = %.2e, min error = %.2e\n",
                s, maximum(errors), minimum(errors))
        scatterlines!(ax3, nsteps_vec, errors, color=colors[i],
            marker=markers_damped[j])
    end
    println()
end

# Reference lines (all dotted grey)
lines!(ax3, nsteps_range, 1e1 ./ (nsteps_range .^ 2),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax3, nsteps_range, 1e-1 ./ (nsteps_range .^ 4),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax3, nsteps_range, 1e-3 ./ (nsteps_range .^ 6),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax3, nsteps_range, 1e-5 ./ (nsteps_range .^ 8),
    linestyle=:dot, color=:gray, linewidth=2)

# Apply shared y-limits across all axes
ylims!(ax1, (1e-15, 1e0))
ylims!(ax2, (1e-15, 1e0))
ylims!(ax3, (1e-15, 1e0))

rowsize!(fig.layout, 1, Relative(2/5))
rowsize!(fig.layout, 2, Relative(3/5))
colsize!(fig.layout, 1, Auto(1))
colsize!(fig.layout, 2, Auto(2))

# =============================================================================
# Shared legend (for Cases 2 & 3)
# =============================================================================

s_entries = [PolyElement(color=colors[i]) for (i, _) in enumerate(s_values)]
s_labels = ["s=$s" for s in s_values]

marker_entries = [MarkerElement(color=:black, marker=m, markersize=12) for m in markers]
Δω_labels = ["Δω=$Δω" for Δω in detunings]

ref_entries = [LineElement(color=:gray, linestyle=:dot, linewidth=2) for _ in 1:4]
ref_labels = [L"O(\Delta t^2)", L"O(\Delta t^4)", L"O(\Delta t^6)", L"O(\Delta t^8)"]

Legend(fig[1:2, 3],
    [s_entries, marker_entries, ref_entries],
    [s_labels, Δω_labels, ref_labels],
    ["Order s", "Δω", "Reference"])

# =============================================================================
# Save
# =============================================================================

println("=" ^ 70)
println("Generating summary plot...")
println("=" ^ 70)

mkpath(joinpath(@__DIR__, "..", "Plots"))
save(joinpath(@__DIR__, "..", "Plots", "dahlquist_experiments.png"), fig)
save(joinpath(@__DIR__, "..", "Plots", "dahlquist_experiments.svg"), fig)
save(joinpath(@__DIR__, "..", "Plots", "dahlquist_experiments.pdf"), fig)
println("Saved figure to Plots/dahlquist_experiments.png")

display(fig)
