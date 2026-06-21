# =============================================================================
# Rabi oscillator convergence study, in three frames × four methods
# =============================================================================
#
# Two-level (Rabi) system  dψ/dt = A(t) ψ = -i H(t) ψ, with
#
#   H_lab(t)   = (ω₀/2) σz + (E/2) cos(ω t) σx                       (lab frame)
#
# transformed to the frame rotating at the drive frequency ω, R = e^{-i(ω/2)σz t},
# giving  H̃ = (δ/2) σz + (E/4) σx + (E/4) cos(2ω t) σx - (E/4) sin(2ω t) σy,
# with detuning δ = ω₀ - ω.  Two further frames come from H̃:
#
#   H_rwa(t)   = (δ/2) σz + (E/4) σx                                 (rotating + RWA)
#   H_norwa(t) = H̃                                                   (rotating, no RWA)
#
# Four methods, each at orders s = 0,1,2 (orders 2/4/6):
#   Hermite            ω_ansatz = 0                      filon_solve_hardcoded
#   Filon              ω_ansatz = drift carrier          filon_solve_hardcoded
#   Controlled Hermite carriers folded as scalars        efficient_controlled_hermite_solve
#   Controlled Filon   carriers factored (ω+ν)           efficient_controlled_filon_solve
#
# Reference: Vern9 at abstol=reltol=1e-13 on each frame's own ODE.
#
# Run:  julia --project=. scripts/rabi_oscillator/rabi_frames_convergence.jl
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using FilonResearch
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEqVerner
using CairoMakie

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())
const inch = 96

const σx = SMatrix{2,2,ComplexF64}(0, 1, 1, 0)
const σy = SMatrix{2,2,ComplexF64}(0, im, -im, 0)   # [0 -i; i 0]
const σz = SMatrix{2,2,ComplexF64}(1, 0, 0, -1)

# -----------------------------------------------------------------------------
# Frame definitions: each returns (co_fourier, co_carrier, ω_ansatz, A_ref)
#   co_fourier — Hermite / Filon / Controlled Hermite (carrier as Fourier control)
#   co_carrier — Controlled Filon (carriers grouped per matrix as a SumControl)
# -----------------------------------------------------------------------------

function frame_lab(ω₀, ω, E)
    driftM = -im * (ω₀ / 2) * σz
    driveM = -im * (E / 2) * σx
    cof = ControlledOperator(
        (ConstantControl(1.0), FourierControl(0.0, [1.0], [0.0], ω)),  # 1, cos(ωt)
        (driftM, driveM))
    coc = ControlledOperator(
        (ConstantControl(1.0),
         SumControl(CarrierControl(ConstantControl(0.5), ω),
                    CarrierControl(ConstantControl(0.5), -ω))),        # ½e^{iωt}+½e^{-iωt}
        (driftM, driveM))
    Aref(t) = -im * ((ω₀ / 2) * σz + (E / 2) * cos(ω * t) * σx)
    return cof, coc, [-ω₀ / 2, ω₀ / 2], Aref
end

function frame_rwa(ω₀, ω, E)
    δ = ω₀ - ω
    driftM = -im * (δ / 2) * σz
    driveM = -im * (E / 4) * σx
    cof = ControlledOperator((ConstantControl(1.0), ConstantControl(1.0)), (driftM, driveM))
    coc = ControlledOperator((ConstantControl(1.0), ConstantControl(1.0)), (driftM, driveM))
    Aref(t) = -im * ((δ / 2) * σz + (E / 4) * σx)
    return cof, coc, [-δ / 2, δ / 2], Aref
end

function frame_norwa(ω₀, ω, E)
    δ = ω₀ - ω
    Az = -im * (δ / 2) * σz
    Ax = -im * σx
    Ay = -im * σy
    cof = ControlledOperator(
        (ConstantControl(1.0),
         FourierControl(E / 4, [E / 4], [0.0], 2ω),      # (E/4)(1 + cos 2ωt)
         FourierControl(0.0, [0.0], [-E / 4], 2ω)),      # -(E/4) sin 2ωt
        (Az, Ax, Ay))
    coc = ControlledOperator(
        (ConstantControl(1.0),
         SumControl(ConstantControl(E / 4 + 0im),
                    CarrierControl(ConstantControl(E / 8 + 0im), 2ω),
                    CarrierControl(ConstantControl(E / 8 + 0im), -2ω)),
         SumControl(CarrierControl(ConstantControl(im * E / 8), 2ω),
                    CarrierControl(ConstantControl(-im * E / 8), -2ω))),
        (Az, Ax, Ay))
    Aref(t) = -im * ((δ / 2) * σz + (E / 4) * σx
                     + (E / 4) * cos(2ω * t) * σx - (E / 4) * sin(2ω * t) * σy)
    return cof, coc, [-δ / 2, δ / 2], Aref
end

# -----------------------------------------------------------------------------
# Solvers and sweep
# -----------------------------------------------------------------------------

const METHODS = (:Hermite, :Filon, :CHermite, :CFilon)
const METHOD_LABELS = ("Hermite", "Filon", "Controlled Hermite", "Controlled Filon")
const SVALS = (0, 1, 2)

function solve_method(method, cof, coc, ωans, ψ0, Δt, n, s)
    if method === :Hermite
        filon_solve_hardcoded(cof, ψ0, zeros(length(ψ0)), Δt, n, s; save_final_only = true)
    elseif method === :Filon
        filon_solve_hardcoded(cof, ψ0, ωans, Δt, n, s; save_final_only = true)
    elseif method === :CHermite
        efficient_controlled_hermite_solve(cof, ψ0, Δt, n, s; save_final_only = true)
    else # :CFilon
        efficient_controlled_filon_solve(coc, ψ0, ωans, Δt, n, s; save_final_only = true)
    end
end

function vern9_ref(Aref, ψ0, T)
    rhs(u, p, t) = Aref(t) * u
    sol = solve(ODEProblem(rhs, SVector{2,ComplexF64}(ψ0), (0.0, T)), Vern9();
                abstol = 1e-13, reltol = 1e-13, save_everystep = false)
    return Vector{ComplexF64}(sol.u[end])
end

"errors[method_index][s_index] :: Vector — final-time 2-norm error over nsteps_list."
function run_sweep(framebuilder, ω₀, ω, E, ψ0, T, nsteps_list)
    cof, coc, ωans, Aref = framebuilder(ω₀, ω, E)
    uref = vern9_ref(Aref, ψ0, T)
    errors = [[Float64[] for _ in SVALS] for _ in METHODS]
    for (mi, m) in enumerate(METHODS), (si, s) in enumerate(SVALS), n in nsteps_list
        u = solve_method(m, cof, coc, ωans, ψ0, T / n, n, s)
        push!(errors[mi][si], norm(Vector{ComplexF64}(u) .- uref))
    end
    return errors, uref
end

# -----------------------------------------------------------------------------
# Plotting: one figure per frame, three panels (s = 0,1,2), four method curves
# -----------------------------------------------------------------------------

const COLORS = Makie.wong_colors()[1:4]
const MARKERS = (:circle, :rect, :xcross, :diamond)

function make_figure(nsteps_list, errors, titlestr, basename; ylims = (1e-15, 1e1))
    fig = Figure(size = (9inch, 3.6inch), fontsize = 11)
    Label(fig[0, 1:3], titlestr; fontsize = 13, font = :bold, padding = (0, 0, 4, 0))
    ns = collect(Float64, nsteps_list)
    nfine = exp10.(range(log10(first(ns)), log10(last(ns)), length = 120))

    for (si, s) in enumerate(SVALS)
        ax = Axis(fig[1, si];
                  title = L"s=%$(s)\;\;(O(\Delta t^{%$(2(s+1))}))",
                  xlabel = "timesteps", ylabel = si == 1 ? "final 2-norm error" : "",
                  xscale = log10, yscale = log10, limits = (nothing, ylims))
        # O(Δtᵖ) guide, anchored to the Hermite curve's coarsest point
        p = 2 * (s + 1)
        C = errors[1][si][1] * float(first(ns))^p
        lines!(ax, nfine, C ./ (nfine .^ p); color = :gray, linestyle = :dot, linewidth = 1.2)
        for mi in 1:length(METHODS)
            scatterlines!(ax, ns, max.(errors[mi][si], ylims[1]);
                          color = COLORS[mi], marker = MARKERS[mi],
                          markersize = 8, linewidth = 1.5)
        end
    end

    entries = [[LineElement(color = COLORS[i], linewidth = 2,
                            points = Point2f[(0, 0.5), (1, 0.5)]),
                MarkerElement(marker = MARKERS[i], color = COLORS[i], markersize = 8)]
               for i in 1:length(METHODS)]
    Legend(fig[1, 4], entries, collect(METHOD_LABELS); framevisible = true)

    mkpath(plotsdir("rabi_oscillator"))
    save(plotsdir("rabi_oscillator", "$(basename).png"), fig; px_per_unit = 2)
    println("  saved → ", plotsdir("rabi_oscillator", "$(basename).png"))
    return fig
end

# =============================================================================
# Run
# =============================================================================

ω₀ = 10.0
ω  = 10.0            # resonant drive (δ = 0)
E  = 1.0
ψ0 = ComplexF64[1.0, 0.0]
T  = 100 * 2π / ω₀
nsteps_list = [2^k for k in 4:14]

frames = (
    ("Lab frame",                 "rabi_frames_lab",    frame_lab),
    ("Rotating frame (RWA)",      "rabi_frames_rwa",    frame_rwa),
    ("Rotating frame (no RWA)",   "rabi_frames_norwa",  frame_norwa),
)

println("=" ^ 70)
println("Rabi convergence: ω₀=$ω₀, ω=$ω, E=$E, T=$(round(T;digits=3)), nsteps=2^$(4:14)")
println("=" ^ 70)

for (title, base, fb) in frames
    println("\n--- $title ---")
    errors, uref = run_sweep(fb, ω₀, ω, E, ψ0, T, nsteps_list)
    for (mi, mlab) in enumerate(METHOD_LABELS)
        for (si, s) in enumerate(SVALS)
            e = errors[mi][si]
            ords = [log2(e[i] / e[i+1]) for i in 1:length(e)-1]
            best = maximum(filter(isfinite, ords); init = -Inf)
            println("  ", rpad(mlab, 19), " s=$s:  ",
                    rpad(string(round(e[1]; sigdigits = 2)), 9), " → ",
                    rpad(string(round(e[end]; sigdigits = 2)), 9),
                    "  best order ≈ ", round(best; digits = 2))
        end
    end
    titlestr = "$title     (ω₀ = $ω₀,  ω = $ω,  E = $E,  δ = $(ω₀ - ω))"
    make_figure(nsteps_list, errors, titlestr, base)
end

println("\nDone.")
