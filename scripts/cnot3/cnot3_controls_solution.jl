# =============================================================================
# CNOT3 control pulses + state evolution (one combined figure)
# =============================================================================
#
# A diagnostic / paper-style figure for the optimized CNOT3 pulse, over a short
# leading window (default 0 ≤ t ≤ 50 ns).  Three stacked panels sharing the time
# axis:
#
#   1. Control Pulses      — Re/Im of the three complex control envelopes
#                            c₁, c₂, c_R (one colour per control, solid = Re,
#                            dashed = Im), in MHz.
#   2. Real Part           — Re of the complex probability amplitudes of the
#   3. Imaginary Part        N most populous states for a chosen initial state.
#
# This combines the paper's "Control Pulses" (Fig. 5) and "Time Evolution of
# Probability Amplitudes" (Fig. 6) figures into one, restricted to the leading
# window.  We deliberately omit the full 0–550 ns evolution: the fast carrier
# oscillations there overlap into a solid band that does not render in
# black-and-white.  Styling follows the SIAM single-column convergence figures
# (cnot3_convergence_paper.jl): latex fonts, thin lines.
#
# The state evolution is an independent high-accuracy Vern9 integration of the
# same A(t) generator the Filon/Hermite solvers use (see cnot3_reference.jl); the
# integrator choice does not matter at this accuracy.  The controls are evaluated
# directly from the QGD CarrierControls, so they include the carrier modulation.
#
# Run:  julia --project=. scripts/cnot3/cnot3_controls_solution.jl
#   ENV overrides: CNOT3_FRAME (rwa|norwa|lab, default rwa),
#                  CNOT3_TWINDOW (window end in ns, default 50),
#                  CNOT3_NSTATES (number of states to plot, default 6).
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using FilonResearch, QuantumGateDesign
using LinearAlgebra: mul!
using OrdinaryDiffEqVerner
using CairoMakie

include(srcdir("cnot3_run.jl"))   # cnot3_hoho_qgd_prob, _controls_and_pcof, adapters

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
const FRAME   = Symbol(get(ENV, "CNOT3_FRAME", "rwa"))
const TWINDOW = parse(Float64, get(ENV, "CNOT3_TWINDOW", "50"))
const NSTATES = parse(Int, get(ENV, "CNOT3_NSTATES", "6"))

# Initial computational state |s b a⟩ (storage, qubit b, qubit a).  The paper's
# Fig. 6 starts from |011⟩.
const INIT_SBA = (0, 1, 1)

# Subsystem sizes (storage, qubit b, qubit a); the last subsystem varies fastest
# in the linear index, so |s b a⟩ ↦ 16 s + 4 b + a + 1.
const NB = 4
const NA = 4
state_index(s, b, a) = NA * NB * s + NA * b + a + 1
function state_label(i)
    i0 = i - 1
    s, b, a = i0 ÷ (NA * NB), (i0 ÷ NA) % NB, i0 % NA
    return L"|%$s%$b%$a\rangle"
end

# Ordinary-frequency MHz per (rad/ns).  The QGD control amplitudes are angular
# frequencies (rad/ns); divide by 2π for cycles and scale ns→MHz.  Set to 1e3 to
# instead report the angular amplitude in "MHz" (the common rad/ns×10³ convention
# seen in some pulse plots).
const MHZ_PER_RADNS = 1e3 / (2π)

# Paper sizing: SIAM single column \linewidth = 5.125 in, 1 pt = 1/72 in.
const PAPER_PT_PER_IN = 72
const PAPER_WIDTH_IN  = 5.125

# -----------------------------------------------------------------------------
# Solve: high-accuracy Vern9 integration of dψ/dt = A(t)ψ over [0, TWINDOW]
# -----------------------------------------------------------------------------
# The Hamiltonian is window-independent; only the integration interval shrinks.
# The controls always use their full 550 ns definition (no squeezing).
function solve_evolution(frame, twindow; npoints = 4000, abstol = 1e-12, reltol = 1e-12)
    qgd_prob = cnot3_hoho_qgd_prob(; frame, Tmax = twindow)
    controls, pcof = cnot3_hoho_controls_and_pcof(; frame)
    co = qgd_to_controlled_operator(qgd_prob, controls, pcof)   # A(t) = Σ cₖ(t) Aₖ

    op = Operator(co, 0.0)                                      # reusable A(t) buffer
    rhs! = (du, u, p, t) -> (evaluate!(op, co, t); mul!(du, op, u); nothing)

    u0 = zeros(ComplexF64, qgd_prob.N_tot_levels)
    u0[state_index(INIT_SBA...)] = 1.0

    tsave = collect(range(0, twindow, length = npoints))
    sol = solve(ODEProblem(rhs!, u0, (0.0, twindow)), Vern9();
                abstol, reltol, saveat = tsave, maxiters = 10_000_000)
    length(sol.t) == length(tsave) ||
        error("Vern9 stopped early ($(length(sol.t)) of $(length(tsave)) saves)")
    H = reduce(hcat, (Vector{ComplexF64}(u) for u in sol.u))
    return tsave, H, controls, pcof
end

# Re/Im of the three complex control envelopes on the save grid, in MHz.
function control_traces(controls, pcof, tsave)
    P = Vector{Vector{Float64}}(undef, length(controls))
    Q = Vector{Vector{Float64}}(undef, length(controls))
    for k in eachindex(controls)
        pk = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
        P[k] = [QuantumGateDesign.eval_p_derivative(controls[k], t, pk, 0) * MHZ_PER_RADNS for t in tsave]
        Q[k] = [QuantumGateDesign.eval_q_derivative(controls[k], t, pk, 0) * MHZ_PER_RADNS for t in tsave]
    end
    return P, Q
end

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
function make_figure(tsave, H, P, Q; basename = "cnot3_controls_solution_$(FRAME)")
    # Rank states by peak amplitude over the window; the initial state peaks at 1.
    peak  = vec(maximum(abs, H; dims = 2))
    order = sortperm(peak; rev = true)
    states = order[1:min(NSTATES, length(order))]

    ccols = Makie.wong_colors()                 # one colour per control
    scols = Makie.wong_colors()                 # one colour per state
    ctrl_lab = (L"c_1", L"c_2", L"c_R")

    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    fig = Figure(size = (W, 420), fontsize = 8, figure_padding = (3, 4, 2, 2))
    ic = L"|%$(INIT_SBA[1])%$(INIT_SBA[2])%$(INIT_SBA[3])\rangle"
    Label(fig[1, 1], L"\textrm{CNOT3\;control\;pulses\;and\;state\;evolution}\;\;(\textrm{%$(FRAME)\;frame},\;%$ic)";
          fontsize = 10, font = :bold)

    # Panel 1: control pulses.
    axc = Axis(fig[2, 1]; ylabel = "Amplitude (MHz)", title = "Control Pulses")
    for k in eachindex(P)
        lines!(axc, tsave, P[k]; color = ccols[k], linewidth = 0.8,
               label = L"\mathrm{Re}\,%$(ctrl_lab[k])")
        lines!(axc, tsave, Q[k]; color = ccols[k], linewidth = 0.8, linestyle = :dash,
               label = L"\mathrm{Im}\,%$(ctrl_lab[k])")
    end
    Legend(fig[3, 1], axc; orientation = :horizontal, framevisible = true,
           labelsize = 7, nbanks = 1, patchsize = (16f0, 6f0), colgap = 6,
           padding = (3f0, 3f0, 2f0, 2f0))

    # Panels 2–3: real and imaginary parts of the populous states' amplitudes.
    axr = Axis(fig[4, 1]; ylabel = "Real Part", title = "Probability Amplitudes")
    axi = Axis(fig[5, 1]; ylabel = "Imaginary Part", xlabel = "Time (nanoseconds)")
    for (j, i) in enumerate(states)
        col = scols[mod1(j, length(scols))]
        lines!(axr, tsave, real.(H[i, :]); color = col, linewidth = 0.6)
        lines!(axi, tsave, imag.(H[i, :]); color = col, linewidth = 0.6,
               label = state_label(i))
    end
    Legend(fig[6, 1], axi; orientation = :horizontal, framevisible = true,
           labelsize = 7, nbanks = 1, patchsize = (14f0, 6f0), colgap = 6)

    for ax in (axc, axr)
        hidexdecorations!(ax, grid = false)
    end
    linkxaxes!(axc, axr, axi)
    xlims!(axi, tsave[1], tsave[end])

    rowgap!(fig.layout, 4)
    rowsize!(fig.layout, 2, Relative(0.26))     # give the control panel some height

    mkpath(plotsdir("cnot3"))
    for (ext, kw) in (("pdf", (; pt_per_unit = 1)), ("png", (; px_per_unit = 3)))
        save(plotsdir("cnot3", "$(basename).$(ext)"), fig; kw...)
    end
    println("  saved → ", plotsdir("cnot3", "$(basename).{pdf,png}"))
    return fig
end

println("Solving CNOT3 evolution: frame=$FRAME, window=[0, $TWINDOW] ns, init=|$(INIT_SBA[1])$(INIT_SBA[2])$(INIT_SBA[3])⟩")
tsave, H, controls, pcof = solve_evolution(FRAME, TWINDOW)
P, Q = control_traces(controls, pcof, tsave)
fig = make_figure(tsave, H, P, Q)
println("Done.")
fig
