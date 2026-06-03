# =============================================================================
# Rabi oscillator convergence study
# =============================================================================
#
# Compares three integrators on the lab-frame Rabi oscillator
#
#     dψ/dt = A(t) ψ,   A(t) = -i [ (ω₀/2) σz + (E/2) cos(ω t) σx ],
#
# for orders s = 0, 1, 2 (methods of order 2, 4, 6):
#
#   1. Hermite           — Filon with zero ansatz frequencies (ω_ansatz = 0)
#   2. Filon             — ansatz frequencies ±ω₀/2 (the drift carrier)
#   3. Controlled-Filon  — additionally factors the drive carrier e^{±iωt} out,
#                          via two CarrierControls for cos(ωt) = ½(e^{iωt}+e^{-iωt})
#
# Each factors out more oscillation, so at fixed order the error should rank
# controlled-Filon < Filon < Hermite.  A second experiment turns the drive off
# (E = 0): the solution is then a constant envelope times the drift carrier, so
# Filon and controlled-Filon are exact to machine precision while Hermite still
# only converges.
#
# Reference solution:  Vern9 at abstol = reltol = 1e-13 for E ≠ 0 (no closed
# form in the lab frame); the exact closed form for E = 0.
#
# Run:  julia --project=FilonExperiments \
#         FilonExperiments/scripts/rabi_oscillator/rabi_filon_convergence.jl
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using FilonResearch                 # ControlledOperator, controls, the two solvers
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEqVerner          # Vern9 (+ ODEProblem/solve via SciMLBase)
using CairoMakie

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())
const inch = 96

# Pauli matrices as static complex matrices.
const σz = SMatrix{2,2,ComplexF64}(1, 0, 0, -1)
const σx = SMatrix{2,2,ComplexF64}(0, 1, 1, 0)

# -----------------------------------------------------------------------------
# Problem setup
# -----------------------------------------------------------------------------

"""
Static (SMatrix) terms of A(t) = -i[(ω₀/2)σz + (E/2)cos(ωt)σx].
`driftM = -i(ω₀/2)σz`, `driveM = -i(E/2)σx`.
"""
function rabi_matrices(ω₀, E)
    driftM = -im * (ω₀ / 2) * σz
    driveM = -im * (E / 2) * σx
    return driftM, driveM
end

"""
The two ControlledOperators used by the three methods.

`co_fourier` (used by Hermite and Filon) carries the drive as the Fourier control
`cos(ωt)`.  `co_carrier` (used by controlled-Filon) splits the drive carrier into
two CarrierControls so the solver can factor `e^{±iωt}` out per term.
"""
function rabi_operators(ω₀, ω, E)
    driftM, driveM = rabi_matrices(ω₀, E)

    co_fourier = ControlledOperator(
        (ConstantControl(1.0), FourierControl(0.0, [1.0], [0.0], ω)),   # 1, cos(ωt)
        (driftM, driveM),
    )

    co_carrier = ControlledOperator(
        (ConstantControl(1.0),
         CarrierControl(ConstantControl(0.5),  ω),                      # ½ e^{+iωt}
         CarrierControl(ConstantControl(0.5), -ω)),                     # ½ e^{-iωt}
        (driftM, driveM, driveM),
    )

    return co_fourier, co_carrier
end

"Independent high-accuracy reference (Vern9) for the final state at time `T`."
function vern9_reference(ω₀, ω, E, ψ0, T)
    A(t) = -im * ((ω₀ / 2) * σz + (E / 2) * cos(ω * t) * σx)
    rhs(u, p, t) = A(t) * u
    u0 = SVector{2,ComplexF64}(ψ0)
    sol = solve(ODEProblem(rhs, u0, (0.0, T)), Vern9();
                abstol = 1e-13, reltol = 1e-13, save_everystep = false)
    return Vector{ComplexF64}(sol.u[end])
end

"Exact final state for the undriven (E = 0) problem: ψ(t) = [e^{-iω₀t/2}ψ_a, e^{+iω₀t/2}ψ_b]."
function exact_reference_E0(ω₀, ψ0, T)
    return ComplexF64[cis(-ω₀ * T / 2) * ψ0[1], cis(ω₀ * T / 2) * ψ0[2]]
end

# -----------------------------------------------------------------------------
# Convergence sweep
# -----------------------------------------------------------------------------

const METHODS = ("Hermite", "Filon", "Controlled-Filon")
const SVALS = (0, 1, 2)

"""
Run all three methods × s ∈ {0,1,2} over `nsteps_list`, returning final-time
2-norm errors as `errors[method_index][s_index] :: Vector{Float64}`.
"""
function run_sweep(ω₀, ω, E, ψ0, T, nsteps_list, uref)
    co_fourier, co_carrier = rabi_operators(ω₀, ω, E)
    ωzero = [0.0, 0.0]
    ωans  = [-ω₀ / 2, ω₀ / 2]

    # method index → (operator, ansatz freqs, solver)
    configs = (
        (co_fourier, ωzero, filon_solve_hardcoded),   # Hermite
        (co_fourier, ωans,  filon_solve_hardcoded),   # Filon
        (co_carrier, ωans,  controlled_filon_solve),  # Controlled-Filon
    )

    errors = [[Float64[] for _ in SVALS] for _ in METHODS]
    for (mi, (co, freqs, solver)) in enumerate(configs)
        for (si, s) in enumerate(SVALS)
            for n in nsteps_list
                u = solver(co, ψ0, freqs, T / n, n, s; save_final_only = true)
                push!(errors[mi][si], norm(Vector{ComplexF64}(u) .- uref))
            end
        end
    end
    return errors
end

"Empirical convergence orders log2(eᵢ/eᵢ₊₁) for a halving stepsize sequence."
empirical_orders(errs) = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]

function report_orders(errors, nsteps_list)
    for (mi, mname) in enumerate(METHODS)
        println("  ", rpad(mname, 18))
        for (si, s) in enumerate(SVALS)
            errs = errors[mi][si]
            ords = empirical_orders(errs)
            finite = filter(isfinite, ords)
            best = isempty(finite) ? NaN : maximum(finite)
            println("    s=$s (order $(2(s+1))):  err ",
                    rpad(string(round(errs[1]; sigdigits = 3)), 11), " → ",
                    rpad(string(round(errs[end]; sigdigits = 3)), 11),
                    "  best empirical order ≈ ", round(best; sigdigits = 3))
        end
    end
end

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

const METHOD_COLORS = Makie.wong_colors()[1:3]
const ORDER_MARKERS = (:circle, :rect, :diamond)

"""
Single combined log-log panel: 9 curves (3 methods × 3 orders), colour = method,
marker = order s, with dotted-grey O(Δtᵖ) guide lines anchored to the Hermite
curves.  Saves png/svg/pdf to `plotsdir()`.
"""
function make_figure(nsteps_list, errors, titlestr, basename; ylims)
    fig = Figure(size = (7inch, 4.4inch), fontsize = 11)
    Label(fig[0, 1:2], titlestr; fontsize = 13, padding = (0, 0, 6, 0))

    ax = Axis(fig[1, 1];
              xlabel = "Number of timesteps",
              ylabel = "Final-time 2-norm error",
              xscale = log10, yscale = log10,
              limits = (nothing, ylims))

    nfine = exp10.(range(log10(first(nsteps_list)), log10(last(nsteps_list)), length = 200))

    # O(Δtᵖ) guide lines, anchored to the Hermite curve of matching order.
    for (si, s) in enumerate(SVALS)
        p = 2 * (s + 1)
        anchor = errors[1][si][1]
        C = anchor * float(first(nsteps_list))^p
        lines!(ax, nfine, C ./ (nfine .^ p);
               color = :gray, linestyle = :dot, linewidth = 1.5)
    end

    for (mi, _) in enumerate(METHODS)
        for (si, _) in enumerate(SVALS)
            scatterlines!(ax, collect(nsteps_list), errors[mi][si];
                          color = METHOD_COLORS[mi], marker = ORDER_MARKERS[si],
                          markersize = 9, linewidth = 1.5)
        end
    end

    method_entries = [LineElement(color = METHOD_COLORS[i], linewidth = 2) for i in 1:3]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[j], color = :black, markersize = 9)
                      for j in 1:3]
    ref_entry      = [LineElement(color = :gray, linestyle = :dot, linewidth = 1.5)]

    Legend(fig[1, 2],
           [method_entries, order_entries, ref_entry],
           [collect(METHODS),
            [L"s=0\;(O(\Delta t^2))", L"s=1\;(O(\Delta t^4))", L"s=2\;(O(\Delta t^6))"],
            [L"O(\Delta t^p)\;\textrm{guide}"]],
           ["Method", "Order", "Slope"],
           orientation = :vertical, tellheight = false, tellwidth = true)

    colsize!(fig.layout, 1, Relative(0.74))

    mkpath(plotsdir())
    for ext in ("png", "svg", "pdf")
        save(plotsdir("$(basename).$(ext)"), fig)
    end
    println("  saved → ", plotsdir("$(basename).{png,svg,pdf}"))
    return fig
end

# =============================================================================
# Experiment 1 — driven Rabi oscillator (E ≠ 0)
# =============================================================================

println("=" ^ 70)
println("Experiment 1: driven Rabi oscillator (E ≠ 0)")
println("=" ^ 70)

ω₀ = 10.0          # transition frequency
ω  = 10.0          # drive frequency (resonant)
E  = 1          # drive strength (E/ω₀ = 0.1, Rabi frequency Ω_R = E/2 = 0.5)
ψ0 = ComplexF64[1.0, 0.0]
T  = 100 * 2π / ω₀
nsteps_list = [2^k for k in 4:14]

println("ω₀=$ω₀, ω=$ω, E=$E, T=2π, ψ0=$(ψ0), nsteps=2^$(4:14)")
println("Computing Vern9 reference...")
uref = vern9_reference(ω₀, ω, E, ψ0, T)
println("Reference final state: ", round.(uref; sigdigits = 6))

errors = run_sweep(ω₀, ω, E, ψ0, T, nsteps_list, uref)
println("\nConvergence (final-time 2-norm error):")
report_orders(errors, nsteps_list)

titlestr = L"\textrm{Rabi\;oscillator\;convergence}\;\;(\omega_0=%$(ω₀).\;\omega=%$(ω),\;E=%$(E))"
fig1 = make_figure(nsteps_list, errors, titlestr, "rabi_filon_convergence";
                   ylims = (1e-14, 1e1))

# =============================================================================
# Experiment 2 — sanity test, undriven (E = 0): Filon family is exact
# =============================================================================

println()
println("=" ^ 70)
println("Experiment 2: sanity test, undriven (E = 0)")
println("=" ^ 70)

E0  = 0.0
ψ0_sanity = ComplexF64[1.0, 1.0] / sqrt(2)   # both carriers exercised
println("ω₀=$ω₀, ω=$ω, E=0, T=2π, ψ0=[1,1]/√2")

uref0 = exact_reference_E0(ω₀, ψ0_sanity, T)
errors0 = run_sweep(ω₀, ω, E0, ψ0_sanity, T, nsteps_list, uref0)
println("\nConvergence (final-time 2-norm error):")
report_orders(errors0, nsteps_list)

# Filon (method 2) and controlled-Filon (method 3) must be exact for all s, all Δt
# — i.e. pinned to the roundoff floor, which accumulates as ~N·eps over the sweep
# (N_max·eps ≈ 16384·2.2e-16 ≈ 4e-12), never showing a convergence slope.
const SANITY_TOL = 1e-10
filon_max  = maximum(maximum.(errors0[2]))
cfilon_max = maximum(maximum.(errors0[3]))
println("\nMax Filon error            = ", round(filon_max;  sigdigits = 3))
println("Max controlled-Filon error = ", round(cfilon_max; sigdigits = 3))
if filon_max < SANITY_TOL && cfilon_max < SANITY_TOL
    println("E=0 SANITY CHECK: PASS (Filon & controlled-Filon at the roundoff floor, < $(SANITY_TOL))")
else
    println("E=0 SANITY CHECK: FAIL")
end

titlestr0 = L"\textrm{Undriven\;sanity\;test\;}(E=0)\textrm{,\;Filon\;family\;exact}"
fig2 = make_figure(nsteps_list, errors0, titlestr0, "rabi_filon_E0_sanity";
                   ylims = (1e-17, 1e1))

println("\nDone.")

fig1

function vern9_reference_full(ω₀, ω, E, ψ0, T)
    A(t) = -im * ((ω₀ / 2) * σz + (E / 2) * cos(ω * t) * σx)
    rhs(u, p, t) = A(t) * u
    u0 = SVector{2,ComplexF64}(ψ0)
    sol = solve(ODEProblem(rhs, u0, (0.0, T)), Vern9();
                abstol = 1e-13, reltol = 1e-13, saveat = 0.01)
    return sol
end

vern9_solution = vern9_reference_full(ω₀, ω, E, ψ0, T)
vern9_matsol = reduce(hcat, vern9_solution.u)
tlist = vern9_solution.t

fig3 = lines

fig3 = Figure(size = (7inch, 4.4inch), fontsize = 11)
ax = Axis(fig3[1, 1];
    xlabel = "t",
              ylabel = "State",
              limits = (nothing, nothing)
)
lines!(ax, tlist, real(vern9_matsol[1,:]))
lines!(ax, tlist, imag(vern9_matsol[1,:]))
lines!(ax, tlist, real(vern9_matsol[2,:]))
lines!(ax, tlist, imag(vern9_matsol[2,:]))
fig3

