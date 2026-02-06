"""
Numerical experiments on the Rabi oscillation problem.

Two-level system driven by a periodic field:
    du/dt = A(t) u

Lab frame (exact):
    A_lab(t) = -i [(ω₀/2)σz + Ω cos(ωt) σx]
             = -i [ω₀/2        Ω cos(ωt)]
                  [Ω cos(ωt)      -ω₀/2 ]

Rotating frame (RWA):
    A_rot = -i [(Δ/2)σz + (Ω/2)σx]
          = -i [Δ/2    Ω/2]
               [Ω/2   -Δ/2]
    where Δ = ω₀ - ω is the detuning.

Experiments:
1. Numerical error in lab frame (compare to true lab frame solution)
2. Numerical error in rotating frame (compare to exact RWA solution)
3. Rotating frame solutions vs lab frame truth (shows RWA error floor)
4. Norm preservation of state vector over time
"""

using FilonResearch
using LinearAlgebra
using CairoMakie
using LaTeXStrings
using Printf

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

# =============================================================================
# Problem setup
# =============================================================================

"""
Lab frame Hamiltonian matrix A(t) = -iH(t)/ℏ for the driven two-level system.
"""
function A_lab(t, ω₀, Ω, ω)
    return -im * [ω₀/2       Ω*cos(ω*t);
                  Ω*cos(ω*t)    -ω₀/2]
end

"""
Time derivative of lab frame A matrix.
"""
function dA_lab(t, ω₀, Ω, ω)
    return -im * [0             -Ω*ω*sin(ω*t);
                  -Ω*ω*sin(ω*t)            0]
end

"""
Second time derivative of lab frame A matrix.
"""
function d2A_lab(t, ω₀, Ω, ω)
    return -im * [0               -Ω*ω^2*cos(ω*t);
                  -Ω*ω^2*cos(ω*t)              0]
end

"""
Third time derivative of lab frame A matrix.
"""
function d3A_lab(t, ω₀, Ω, ω)
    return -im * [0               Ω*ω^3*sin(ω*t);
                  Ω*ω^3*sin(ω*t)              0]
end

"""
Rotating frame (RWA) Hamiltonian matrix - time independent.
"""
function A_rot(Δ, Ω)
    return -im * [Δ/2   Ω/2;
                  Ω/2  -Δ/2]
end

"""
Exact solution in the rotating frame (RWA).
Ω_eff = √(Δ² + Ω²) is the generalized Rabi frequency.
"""
function exact_rotating_frame(u0, Δ, Ω, t)
    A = A_rot(Δ, Ω)
    return exp(A * t) * u0
end

"""
Transform state from rotating frame to lab frame.
U(t) = diag(e^{-iωt/2}, e^{iωt/2})
ψ_lab(t) = U(t) ψ_rot(t)
"""
function rotating_to_lab(u_rot, ω, t)
    U = [cis(-ω*t/2)  0;
         0            cis(ω*t/2)]
    return U * u_rot
end

"""
Transform state from lab frame to rotating frame.
U†(t) = diag(e^{iωt/2}, e^{-iωt/2})
ψ_rot(t) = U†(t) ψ_lab(t)
"""
function lab_to_rotating(u_lab, ω, t)
    U_dag = [cis(ω*t/2)   0;
             0            cis(-ω*t/2)]
    return U_dag * u_lab
end

"""
Compute a Filon-based lab frame reference solution at nsteps+1 evenly spaced points.
Uses filon_solve with exponential integration (ansatz frequencies) and s=s_ref for high accuracy.
"""
function lab_reference_filon(A_deriv_funcs, u0, frequencies, tf, nsteps, s_ref)
    return filon_solve(A_deriv_funcs, u0, frequencies, tf, nsteps, s_ref)
end

"""
Compute a saturated Filon-based reference solution by doubling nsteps until convergence.
Returns (nsteps_ref, sol_ref) where sol_ref is a (d × nsteps_ref+1) matrix.
"""
function saturated_lab_reference(A_deriv_funcs, u0, frequencies, tf, s_ref;
                                  nsteps_start=2^14, nsteps_max=2^20, tol=1e-13)
    nsteps_prev = nsteps_start
    sol_prev = lab_reference_filon(A_deriv_funcs, u0, frequencies, tf, nsteps_prev, s_ref)

    nsteps_curr = 2 * nsteps_prev
    while nsteps_curr <= nsteps_max
        sol_curr = lab_reference_filon(A_deriv_funcs, u0, frequencies, tf, nsteps_curr, s_ref)
        # Compare endpoints
        diff = norm(sol_curr[:, end] - sol_prev[:, end])
        println("  nsteps=$nsteps_curr: endpoint diff = $diff")
        if diff < tol
            println("  Saturated at nsteps=$nsteps_curr")
            return nsteps_curr, sol_curr
        end
        nsteps_prev = nsteps_curr
        sol_prev = sol_curr
        nsteps_curr *= 2
    end

    println("  Warning: did not saturate by nsteps=$nsteps_prev, using that as reference")
    return nsteps_prev, sol_prev
end

"""
Subsample the reference solution matrix to match a coarser grid with nsteps_test steps.
Returns a length-(nsteps_test+1) vector of d-vectors.
"""
function subsample_reference(sol_ref, nsteps_ref, nsteps_test)
    ratio = nsteps_ref ÷ nsteps_test
    return [sol_ref[:, 1 + (k-1)*ratio] for k in 1:nsteps_test+1]
end

"""
Compute the discrete L2 norm of the error over [0, tf].
`sols` is a (d × N) matrix (filon_solve output), `refs` is a length-N vector of d-vectors.
Uses ||e||_h = √(h ∑ᵢ ||eᵢ||²) where h = Δt is the grid spacing.
"""
function l2_error(times, sols, refs)
    return sqrt(sum(sum(abs2, sols[:, i] - refs[i]) for i in 1:length(times)) / length(times))
end


# =============================================================================
# Main experiment parameters
# =============================================================================

ω₀ = 10.0           # Atomic transition frequency
ω = 10.0            # Drive frequency (on resonance: ω = ω₀)
Δ = ω₀ - ω          # Detuning
Ω = 1.0             # Rabi frequency (coupling strength)

# Initial state: ground state
u0 = ComplexF64[0.0, 1.0]

# Ansatz frequencies for Filon method (energy eigenvalues of free Hamiltonian)
frequencies_lab = [ω₀/2, -ω₀/2]
frequencies_rot = [0.0, 0.0]  # In rotating frame, effectively removed fast oscillation

# Time parameters
n_periods = 5.5                    # Number of Rabi periods
Ω_eff = sqrt(Δ^2 + Ω^2)          # Generalized Rabi frequency
T_rabi = 2π / Ω_eff              # Rabi period
tf = n_periods * T_rabi          # Final time

s_values = 0:3
use_l2_error = false # false for final-time error only

colors = Makie.wong_colors()

println("=" ^ 70)
println("Rabi Oscillation Experiments")
println("=" ^ 70)
println("ω₀ = $ω₀ (atomic frequency)")
println("ω = $ω (drive frequency)")
println("Δ = $Δ (detuning)")
println("Ω = $Ω (Rabi frequency)")
println("Ω_eff = $Ω_eff (generalized Rabi frequency)")
println("T_rabi = $T_rabi (Rabi period)")
println("tf = $tf ($n_periods Rabi periods)")
println()

# =============================================================================
# Reference solutions
# =============================================================================

println("-" ^ 70)
println("Computing reference solutions...")
println("-" ^ 70)

A_lab_func = t -> A_lab(t, ω₀, Ω, ω)
dA_lab_func = t -> dA_lab(t, ω₀, Ω, ω)
d2A_lab_func = t -> d2A_lab(t, ω₀, Ω, ω)
d3A_lab_func = t -> d3A_lab(t, ω₀, Ω, ω)
A_deriv_funcs_all = (A_lab_func, dA_lab_func, d2A_lab_func, d3A_lab_func)
A_rot_mat = A_rot(Δ, Ω)

frequencies_zero = [0.0, 0.0]
s_ref = 3

# Rotating frame exact solution (analytical)
u_final_rot = exact_rotating_frame(u0, Δ, Ω, tf)

# Transform rotating frame solution to lab frame for proper comparison
u_final_rot_in_lab = rotating_to_lab(u_final_rot, ω, tf)

# High-accuracy lab frame reference (Filon-based, saturated)
println("Saturating lab frame reference...")
nsteps_ref, sol_ref = saturated_lab_reference(A_deriv_funcs_all, u0, frequencies_zero, tf, s_ref)
u_final_lab = sol_ref[:, end]

println("Lab frame final state: $u_final_lab")
println("Lab frame final norm: $(norm(u_final_lab))")
println("Rotating frame final state (in rot frame): $u_final_rot")
println("Rotating frame final state (in lab frame): $u_final_rot_in_lab")
println("RWA error (proper comparison in lab frame): $(norm(u_final_lab - u_final_rot_in_lab))")
println()

inch = 96  # points per inch

# =============================================================================
# Experiment figure
# =============================================================================

fig = Figure(size=(6.5inch, 9inch), fontsize=12)

# Main title
Label(fig[0, 1:2],
    L"\textbf{Filon Method on Rabi Oscillation: } \dot{u} = -i\left(\frac{\omega_0}{2}\sigma_z + \Omega\cos(\omega t)\,\sigma_x\right)u, \quad \omega_0 = %$(Int(ω₀)),\; \Omega = %$(Int(Ω))",
    fontsize=14, padding=(0, 0, 0, 0))

nsteps_range = [2^k for k in 2:13]
nsteps_fine = exp10.(range(log10(first(nsteps_range)), log10(last(nsteps_range)), length=200))

# =============================================================================
# Experiment 1: Numerical error in lab frame
# =============================================================================

println("-" ^ 70)
println("Experiment 1: Numerical error in lab frame")
println("-" ^ 70)

error_label = use_l2_error ? L"L^2 \textrm{ Error}" : L"\textrm{Final time error}"

ax1 = Axis(fig[1, 1],
    title=L"\textrm{Lab frame}",
    titlegap=10,
    xlabel="Number of Timesteps",
    ylabel=error_label,
    xscale=log10, yscale=log10,
    limits=(nothing, (1e-15, 1e0)))

for (i, s) in enumerate(s_values)
    errors = Float64[]
    for nsteps in nsteps_range
        sol = filon_solve(A_deriv_funcs_all, u0, frequencies_lab, tf, nsteps, s)
        if use_l2_error
            times = collect(range(0, tf, length=nsteps+1))
            refs = subsample_reference(sol_ref, nsteps_ref, nsteps)
            push!(errors, l2_error(times, sol, refs))
        else
            push!(errors, norm(sol[:, end] - sol_ref[:, end]))
        end
    end
    scatterlines!(ax1, nsteps_range, errors, color=colors[i],
        marker=:circle)
    @printf("  s=%d: max error = %.2e, min error = %.2e\n", s, maximum(errors), minimum(errors))
end

# Reference lines
lines!(ax1, nsteps_fine, 2e6 ./ (nsteps_fine .^ 2),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax1, nsteps_fine, 4e9 ./ (nsteps_fine .^ 4),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax1, nsteps_fine, 4e12 ./ (nsteps_fine .^ 6),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax1, nsteps_fine, 4e15 ./ (nsteps_fine .^ 8),
    linestyle=:dot, color=:gray, linewidth=2)

println()

# =============================================================================
# Experiment 2: Numerical error in rotating frame
# =============================================================================

println("-" ^ 70)
println("Experiment 2: Numerical error in rotating frame")
println("-" ^ 70)

ax2 = Axis(fig[2, 1],
    title=L"\textrm{Rotating frame}",
    titlegap=10,
    xlabel="Number of Timesteps",
    ylabel=error_label,
    xscale=log10, yscale=log10,
    limits=(nothing, (1e-15, 1e0)))

for (i, s) in enumerate(s_values)
    errors = Float64[]
    for nsteps in nsteps_range
        sol = filon_solve(A_rot_mat, u0, frequencies_rot, tf, nsteps, s)
        if use_l2_error
            times = collect(range(0, tf, length=nsteps+1))
            refs = [exp(A_rot_mat * t) * u0 for t in times]
            push!(errors, l2_error(times, sol, refs))
        else
            push!(errors, norm(sol[:, end] - exp(A_rot_mat * tf) * u0))
        end
    end
    scatterlines!(ax2, nsteps_range, errors, color=colors[i],
        marker=:circle)
    @printf("  s=%d: max error = %.2e, min error = %.2e\n", s, maximum(errors), minimum(errors))
end

# Reference lines
lines!(ax2, nsteps_fine, 1e1 ./ (nsteps_fine .^ 2),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax2, nsteps_fine, 1e1 ./ (nsteps_fine .^ 4),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax2, nsteps_fine, 1e1 ./ (nsteps_fine .^ 6),
    linestyle=:dot, color=:gray, linewidth=2)
lines!(ax2, nsteps_fine, 1e1 ./ (nsteps_fine .^ 8),
    linestyle=:dot, color=:gray, linewidth=2)

println()

# =============================================================================
# Experiment 3: Rotating frame vs lab frame truth
# =============================================================================

println("-" ^ 70)
println("Experiment 3: Rotating frame solutions vs lab frame truth")
println("-" ^ 70)

ax3 = Axis(fig[3, 1],
    title=L"\textrm{Rotating frame vs lab frame truth}",
    titlegap=10,
    xlabel="Number of Timesteps",
    ylabel=error_label,
    xscale=log10, yscale=log10,
    limits=(nothing, (1e-15, 1e0)))

# RWA error (constant floor)
if use_l2_error
    nsteps_rwa = 1024  # power of 2 for clean subsampling
    times_rwa = collect(range(0, tf, length=nsteps_rwa+1))
    rwa_rot_exact = hcat([rotating_to_lab(exp(A_rot_mat * t) * u0, ω, t) for t in times_rwa]...)
    rwa_lab_ref = subsample_reference(sol_ref, nsteps_ref, nsteps_rwa)
    rwa_error = l2_error(times_rwa, rwa_rot_exact, rwa_lab_ref)
else
    rwa_error = norm(rotating_to_lab(exp(A_rot_mat * tf) * u0, ω, tf) - sol_ref[:, end])
end
println("RWA approximation error: $rwa_error")

for (i, s) in enumerate(s_values)
    errors = Float64[]
    for nsteps in nsteps_range
        sol = filon_solve(A_rot_mat, u0, frequencies_rot, tf, nsteps, s)
        if use_l2_error
            times = collect(range(0, tf, length=nsteps+1))
            sol_in_lab = hcat([rotating_to_lab(sol[:, k], ω, times[k]) for k in 1:nsteps+1]...)
            refs = subsample_reference(sol_ref, nsteps_ref, nsteps)
            push!(errors, l2_error(times, sol_in_lab, refs))
        else
            sol_final_in_lab = rotating_to_lab(sol[:, end], ω, tf)
            push!(errors, norm(sol_final_in_lab - sol_ref[:, end]))
        end
    end
    scatterlines!(ax3, nsteps_range, errors, color=colors[i],
        marker=:circle)
    @printf("  s=%d: max error = %.2e, min error = %.2e\n", s, maximum(errors), minimum(errors))
end

if rwa_error > 1e-15
    hlines!(ax3, [rwa_error], linestyle=:dash, color=:black, linewidth=2)
end


println()

# =============================================================================
# Experiment 4: Norm preservation
# =============================================================================

println("-" ^ 70)
println("Experiment 4: Norm preservation")
println("-" ^ 70)

nsteps_norm = 100
t_vec = collect(range(0, tf, length=nsteps_norm+1))

for (i, s) in enumerate(s_values)
    sol = filon_solve(A_deriv_funcs_all, u0, frequencies_lab, tf, nsteps_norm, s)
    norms = [norm(sol[:, k]) for k in 1:size(sol, 2)]
    @printf("  s=%d: initial norm = %.6f, final norm = %.6f, change = %.2e\n",
            s, norms[1], norms[end], norms[end] - norms[1])
end

println()

# =============================================================================
# Shared legend
# =============================================================================

s_entries = [PolyElement(color=colors[i]) for (i, _) in enumerate(s_values)]
s_labels = ["s=$s" for s in s_values]

ref_entries = [
    LineElement(color=:gray, linestyle=:dot, linewidth=2),
    LineElement(color=:gray, linestyle=:dot, linewidth=2),
    LineElement(color=:gray, linestyle=:dot, linewidth=2),
    LineElement(color=:gray, linestyle=:dot, linewidth=2),
    LineElement(color=:black, linestyle=:dash, linewidth=2),
]
ref_labels = [L"O(\Delta t^2)", L"O(\Delta t^4)", L"O(\Delta t^6)", L"O(\Delta t^8)", "RWA error"]

Legend(fig[1:3, 2],
    [s_entries, ref_entries],
    [s_labels, ref_labels],
    ["Order s", "Reference"])

# =============================================================================
# Save experiment figure
# =============================================================================

println("=" ^ 70)
println("Generating experiment plot...")
println("=" ^ 70)

mkpath(joinpath(@__DIR__, "..", "Plots"))
save(joinpath(@__DIR__, "..", "Plots", "rabi_experiments.png"), fig)
save(joinpath(@__DIR__, "..", "Plots", "rabi_experiments.svg"), fig)
save(joinpath(@__DIR__, "..", "Plots", "rabi_experiments.pdf"), fig)
println("Saved figure to Plots/rabi_experiments.{png,svg,pdf}")

display(fig)

# =============================================================================
# Solution plots (lab frame and rotating frame)
# =============================================================================

println()
println("=" ^ 70)
println("Generating solution plots...")
println("=" ^ 70)

nsteps_plot = 512  # power of 2 for clean subsampling
t_plot = collect(range(0, tf, length=nsteps_plot+1))

# Compute lab frame solution using Filon with zero frequencies and s=3
sol_lab_plot = filon_solve(A_deriv_funcs_all, u0, frequencies_zero, tf, nsteps_plot, s_ref)
u_lab_history = [sol_lab_plot[:, k] for k in 1:size(sol_lab_plot, 2)]

# Compute rotating frame solution
sol_rot = filon_solve(A_rot_mat, u0, frequencies_rot, tf, nsteps_plot, 0)

# Extract populations
pop_excited_lab = [abs2(u_lab_history[k][1]) for k in 1:length(u_lab_history)]
pop_ground_lab = [abs2(u_lab_history[k][2]) for k in 1:length(u_lab_history)]

pop_excited_rot = [abs2(sol_rot[1, k]) for k in 1:size(sol_rot, 2)]
pop_ground_rot = [abs2(sol_rot[2, k]) for k in 1:size(sol_rot, 2)]

# Extract complex amplitudes
u1_lab = [u_lab_history[k][1] for k in 1:length(u_lab_history)]
u2_lab = [u_lab_history[k][2] for k in 1:length(u_lab_history)]

u1_rot = [sol_rot[1, k] for k in 1:size(sol_rot, 2)]
u2_rot = [sol_rot[2, k] for k in 1:size(sol_rot, 2)]

# Rotating frame solution in lab frame
u_rot_in_lab = zeros(ComplexF64, 2, length(t_plot))
for k in 1:length(t_plot)
    u_rot_in_lab[:, k] = rotating_to_lab(sol_rot[:, k], ω, t_plot[k])
end

u1_rot_in_lab = u_rot_in_lab[1, :]
u2_rot_in_lab = u_rot_in_lab[2, :]

fig2 = Figure(size=(6.5inch, 7.5inch), fontsize=12)

Label(fig2[0, 1:2],
    L"\textbf{Rabi Oscillation Solutions: } \dot{u} = -i\left(\frac{\omega_0}{2}\sigma_z + \Omega\cos(\omega t)\,\sigma_x\right)u, \quad \omega_0 = %$(Int(ω₀)),\; \Omega = %$(Int(Ω)),\; \omega = %$(Int(ω))",
    fontsize=14, padding=(0, 0, 0, 0))

# --- Lab frame: complex amplitudes ---
ax_la = Axis(fig2[1, 1],
    title=L"\textrm{Lab frame}",
    titlegap=10,
    xlabel="Time",
    ylabel="Amplitude")
lines!(ax_la, t_plot, real.(u1_lab), color=colors[1])
lines!(ax_la, t_plot, imag.(u1_lab), color=colors[2])
lines!(ax_la, t_plot, real.(u2_lab), color=colors[3])
lines!(ax_la, t_plot, imag.(u2_lab), color=colors[4])

# --- Rotating frame: complex amplitudes ---
ax_ra = Axis(fig2[2, 1],
    title=L"\textrm{Rotating frame (RWA)}",
    titlegap=10,
    xlabel="Time",
    ylabel="Amplitude")
lines!(ax_ra, t_plot, real.(u1_rot), color=colors[1])
lines!(ax_ra, t_plot, imag.(u1_rot), color=colors[2])
lines!(ax_ra, t_plot, real.(u2_rot), color=colors[3])
lines!(ax_ra, t_plot, imag.(u2_rot), color=colors[4])

# --- Error: rotating frame (in lab coordinates) vs lab frame ---
err_rot_lab = u_rot_in_lab .- sol_lab_plot
ax_err = Axis(fig2[3, 1],
    title=L"\textrm{Error: rotating frame (in lab) vs lab frame}",
    titlegap=10,
    xlabel="Time",
    ylabel="Error")
lines!(ax_err, t_plot, real.(err_rot_lab[1, :]), color=colors[1])
lines!(ax_err, t_plot, imag.(err_rot_lab[1, :]), color=colors[2])
lines!(ax_err, t_plot, real.(err_rot_lab[2, :]), color=colors[3])
lines!(ax_err, t_plot, imag.(err_rot_lab[2, :]), color=colors[4])

# Shared amplitude legend
amp_entries = [LineElement(color=colors[i]) for i in 1:4]
amp_labels = [L"\textrm{Re}(u_1)", L"\textrm{Im}(u_1)", L"\textrm{Re}(u_2)", L"\textrm{Im}(u_2)"]
Legend(fig2[1:3, 2], amp_entries, amp_labels, "Component")

save(joinpath(@__DIR__, "..", "Plots", "rabi_solutions.png"), fig2)
save(joinpath(@__DIR__, "..", "Plots", "rabi_solutions.svg"), fig2)
save(joinpath(@__DIR__, "..", "Plots", "rabi_solutions.pdf"), fig2)
println("Saved figure to Plots/rabi_solutions.{png,svg,pdf}")

display(fig2)
