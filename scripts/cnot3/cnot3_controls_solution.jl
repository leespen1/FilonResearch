# =============================================================================
# CNOT3 control pulses + state evolution (one combined figure)
# =============================================================================
#
# A paper-style figure for the optimized CNOT3 pulse over a leading window
# (default 0 ≤ t ≤ 50 ns).  Three stacked panels sharing the time axis:
#
#   1. Control Pulses                    — Re/Im of the three complex control
#                                          envelopes c₁, c₂, c_R (MHz/2π).
#   2. Probability Amplitudes (Real Part), Lab Frame
#   3. Probability Amplitudes (Real Part), Rotating Frame
#
# Panels 2–3 show the real part of the complex probability amplitudes of the N
# most populous states (ranked by mean population over the WHOLE 0–550 ns
# trajectory) for a chosen initial state, the same states/colours in both frames.
# The lab frame carries the bare oscillator carriers (states oscillate at several
# GHz), so its curves form a dense band; the rotating frame removes them, leaving
# the slow envelope (cf. the paper's Fig. 6).
#
# State evolution is an independent high-accuracy Vern9 integration of the same
# A(t) generator the Filon/Hermite solvers use (see cnot3_reference.jl); the
# integrator choice does not matter at this accuracy.  Solves are cached as jld2
# under datadir("cnot3_controls_solution") — a SEPARATE directory/prefix from the
# convergence runs, so `collect_results` over the convergence data never sees them.
#
# Run:  julia --project=. scripts/cnot3/cnot3_controls_solution.jl
#   ENV overrides: CNOT3_ROT_FRAME (rwa|norwa, default rwa),
#                  CNOT3_TSTART / CNOT3_TSTOP (display window ns, default 0/50),
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
const ROT_FRAME = Symbol(get(ENV, "CNOT3_ROT_FRAME", "rwa"))   # rotating-frame panel + controls
const LAB_FRAME = :lab
const TSTART  = parse(Float64, get(ENV, "CNOT3_TSTART", "50"))
const TSTOP   = parse(Float64, get(ENV, "CNOT3_TSTOP", "75"))
const NSTATES = parse(Int, get(ENV, "CNOT3_NSTATES", "6"))

# The lab-frame panel uses its own, much narrower window [TSTART, LAB_TSTOP] so its
# multi-GHz carrier oscillations are actually resolved (over [TSTART, TSTOP] they
# would collapse into a solid band).
const LAB_TSTOP = parse(Float64, get(ENV, "CNOT3_LAB_TSTOP", "50.5"))

# Full gate duration: populous-state ranking is over this whole interval.
const TFULL = 550.0
# Rotating-frame solve spans [0, TFULL] (ranking + display); points chosen so the
# display window resolves the slow envelope smoothly (~60 points/ns).
const NPOINTS_FULL = 33_000
# Lab-frame solve spans only [0, LAB_TSTOP], at high density (~320 points/ns) so
# the ~10 GHz carriers are smooth in the narrow lab window.
const NPOINTS_LAB = round(Int, LAB_TSTOP * 320)

# Solver tolerances (plenty for plotting; the lab frame is stiff).
const ABSTOL = 1e-11
const RELTOL = 1e-11

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

# Control amplitude scale.  The QGD control values are angular frequencies
# (rad/ns).  Axis is labelled "MHz/2π": the plotted number is the angular
# amplitude eval_p × 10³ (so a value X means the ordinary frequency is X/(2π)
# MHz).  Divide by an extra 2π (i.e. use 1e3/(2π)) to plot ordinary-frequency MHz.
const CONTROL_SCALE = 1e3
const CONTROL_UNIT_LABEL = L"\textrm{Amplitude}\;(\textrm{MHz}/2\pi)"

# Paper sizing: SIAM single column \linewidth = 5.125 in, 1 pt = 1/72 in.
const PAPER_PT_PER_IN = 72
const PAPER_WIDTH_IN  = 5.125

# -----------------------------------------------------------------------------
# Solve: high-accuracy Vern9 integration of dψ/dt = A(t)ψ over [0, tstop]
# -----------------------------------------------------------------------------
# The Hamiltonian is window-independent; only the integration interval changes.
# The controls always use their full 550 ns definition (no squeezing).
function solve_evolution(frame, tstop, npoints; abstol = ABSTOL, reltol = RELTOL)
    qgd_prob = cnot3_hoho_qgd_prob(; frame, Tmax = tstop)
    controls, pcof = cnot3_hoho_controls_and_pcof(; frame)
    co = qgd_to_controlled_operator(qgd_prob, controls, pcof)   # A(t) = Σ cₖ(t) Aₖ

    op = Operator(co, 0.0)                                      # reusable A(t) buffer
    rhs! = (du, u, p, t) -> (evaluate!(op, co, t); mul!(du, op, u); nothing)

    u0 = zeros(ComplexF64, qgd_prob.N_tot_levels)
    u0[state_index(INIT_SBA...)] = 1.0

    tsave = collect(range(0, tstop, length = npoints))
    sol = solve(ODEProblem(rhs!, u0, (0.0, tstop)), Vern9();
                abstol, reltol, saveat = tsave, maxiters = 50_000_000)
    length(sol.t) == length(tsave) ||
        error("Vern9 stopped early ($(length(sol.t)) of $(length(tsave)) saves)")
    return tsave, stack(sol.u)                                 # 160 × npoints, O(n)
end

# Cached wrapper: results land in a dedicated dir/prefix, NOT the convergence data.
function solve_cached(frame, tstop, npoints; abstol = ABSTOL, reltol = RELTOL)
    cfg = Dict("frame" => string(frame), "init" => join(string.(INIT_SBA)),
               "tstop" => tstop, "npoints" => npoints, "abstol" => abstol, "reltol" => reltol)
    data, _ = produce_or_load(cfg, datadir("cnot3_controls_solution");
                              prefix = "cnot3_ctrlsol", tag = false, verbose = true) do cfg
        tsave, H = solve_evolution(frame, tstop, npoints; abstol, reltol)
        Dict("tsave" => tsave, "H" => H)
    end
    return data["tsave"], data["H"]
end

# Re/Im of the three complex control envelopes on a time grid, in MHz/2π.
function control_traces(controls, pcof, ts)
    P = Vector{Vector{Float64}}(undef, length(controls))
    Q = Vector{Vector{Float64}}(undef, length(controls))
    for k in eachindex(controls)
        pk = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
        P[k] = [QuantumGateDesign.eval_p_derivative(controls[k], t, pk, 0) * CONTROL_SCALE for t in ts]
        Q[k] = [QuantumGateDesign.eval_q_derivative(controls[k], t, pk, 0) * CONTROL_SCALE for t in ts]
    end
    return P, Q
end

window_indices(ts, a, b) = findall(t -> a <= t <= b, ts)

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
function make_figure(; basename = "cnot3_controls_solution")
    # Rotating frame: full-range solve → rank states, plus the rotating-frame panel.
    t_rot, H_rot = solve_cached(ROT_FRAME, TFULL, NPOINTS_FULL)
    meanpop = vec(sum(abs2, H_rot; dims = 2)) ./ size(H_rot, 2)
    states  = sortperm(meanpop; rev = true)[1:min(NSTATES, length(meanpop))]

    # Lab frame: narrow high-density window solve only (same states/colours).
    t_lab, H_lab = solve_cached(LAB_FRAME, LAB_TSTOP, NPOINTS_LAB)

    win_rot = window_indices(t_rot, TSTART, TSTOP)
    win_lab = window_indices(t_lab, TSTART, LAB_TSTOP)
    tw     = t_rot[win_rot]
    t_labw = t_lab[win_lab]
    # Control envelopes per frame on each column's window grid.  In the lab frame the
    # q-part multiplies a zeroed operator, so only the real part (the physical lab
    # pulse) is shown there; the rotating frame shows the full complex envelope.
    P_rwa, Q_rwa = control_traces(cnot3_hoho_controls_and_pcof(; frame = ROT_FRAME)..., tw)
    P_lab, _     = control_traces(cnot3_hoho_controls_and_pcof(; frame = LAB_FRAME)..., t_labw)

    ccols = Makie.wong_colors()
    scols = Makie.wong_colors()
    ctrl  = (L"c_1", L"c_2", L"c_R")
    rot_label = ROT_FRAME === :rwa ? "RWA Frame" : "Rotating Frame"

    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    # Extra right padding so the rightmost x-tick label (e.g. "100") is not clipped.
    fig = Figure(size = (W, 324), fontsize = 8, figure_padding = (3, 8, 2, 2))
    ic = L"|%$(INIT_SBA[1])%$(INIT_SBA[2])%$(INIT_SBA[3])\rangle"
    Label(fig[1, 1:2], L"\textbf{Time\;Evolution\;Under\;CNOT\;Pulse},\;|\psi_0\rangle = %$ic";
          fontsize = 10, font = :bold, tellwidth = false)

    # 2×2 layout: columns = lab (narrow zoom [TSTART, LAB_TSTOP]) vs rotating frame
    # ([TSTART, TSTOP]); rows = probability amplitudes (imaginary part) vs controls.
    # The lab amplitudes span ±1; the rotating-frame envelope is smaller (±0.5).
    lab_amp_ticks = (limits = (nothing, (-1.0, 1.0)), yticks = [-1.0, -0.5, 0.0, 0.5, 1.0],
                     yminorticksvisible = true, yminorticks = [-0.75, -0.25, 0.25, 0.75])
    rwa_amp_ticks = (limits = (nothing, (-0.5, 0.5)), yticks = collect(-0.5:0.25:0.5))
    # Control y-limits chosen to end on major ticks while containing the data
    # (lab real pulse ≈ ±30, rotating envelope ≈ ±14).
    lab_ctrl_ticks = (limits = (nothing, (-40.0, 40.0)), yticks = [-40.0, -20.0, 0.0, 20.0, 40.0],
                      yminorticksvisible = true, yminorticks = Makie.IntervalsBetween(2))
    rwa_ctrl_ticks = (limits = (nothing, (-15.0, 15.0)), yticks = collect(-15.0:5.0:15.0))
    # Shared per-column x ticks: both rows of a column use these, so the amplitude and
    # control panels have identical x gridlines and the window ends on a major tick.
    # The lab column keeps minor ticks; the RWA column has none.
    lab_xt = (xticks = [50.0, 50.25, 50.5],
              xminorticksvisible = true, xminorticks = Makie.IntervalsBetween(5))
    rwa_xt = (xticks = collect(TSTART:5.0:TSTOP),)
    ctrl_ylabel = L"\textrm{Pulse\;Amplitude}\;(\textrm{MHz}/2\pi)"

    # Row 2: probability amplitudes (imaginary part), lab (left) and rotating (right).
    axA_lab = Axis(fig[2, 1]; title = "Lab Frame", ylabel = "Im. Probability Amplitude",
                   lab_amp_ticks..., lab_xt...)
    axA_rwa = Axis(fig[2, 2]; title = rot_label, yaxisposition = :right, rwa_amp_ticks..., rwa_xt...)
    for (j, i) in enumerate(states)
        col = scols[mod1(j, length(scols))]
        lines!(axA_lab, t_labw, imag.(H_lab[i, win_lab]); color = col, linewidth = 0.6)
        lines!(axA_rwa, tw, imag.(H_rot[i, win_rot]); color = col, linewidth = 0.6,
               label = state_label(i))
    end
    Legend(fig[3, 1:2], axA_rwa; orientation = :horizontal, framevisible = true,
           labelsize = 7, nbanks = 1, patchsize = (14f0, 6f0), colgap = 6, tellwidth = false)

    # Row 4: control pulses, lab (real drive only) and rotating (complex envelope).
    # Colours: the three Re traces use the first three palette colours, the three Im
    # traces the next three — so the lab panel's three real pulses are clearly distinct.
    axC_lab = Axis(fig[4, 1]; ylabel = ctrl_ylabel, xlabel = "Time (nanoseconds)",
                   lab_ctrl_ticks..., lab_xt...)
    axC_rwa = Axis(fig[4, 2]; yaxisposition = :right, xlabel = "Time (nanoseconds)",
                   rwa_ctrl_ticks..., rwa_xt...)
    for k in eachindex(P_rwa)
        lines!(axC_lab, t_labw, P_lab[k]; color = ccols[k], linewidth = 0.8)
        lines!(axC_rwa, tw, P_rwa[k]; color = ccols[k], linewidth = 0.8,
               label = L"\mathrm{Re}\,%$(ctrl[k])")
    end
    for k in eachindex(Q_rwa)
        lines!(axC_rwa, tw, Q_rwa[k]; color = ccols[3 + k], linewidth = 0.8,
               label = L"\mathrm{Im}\,%$(ctrl[k])")
    end
    Legend(fig[5, 1:2], axC_rwa; orientation = :horizontal, framevisible = true,
           labelsize = 7, nbanks = 1, patchsize = (16f0, 6f0), colgap = 6,
           padding = (3f0, 3f0, 2f0, 2f0), tellwidth = false)

    # Keep the x tick marks (major + lab minor) on the amplitude panels so they match
    # the control panels below; only the redundant numeric tick labels are hidden.
    hidexdecorations!(axA_lab; ticks = false, minorticks = false, grid = false, minorgrid = false)
    hidexdecorations!(axA_rwa; ticks = false, minorticks = false, grid = false, minorgrid = false)
    linkxaxes!(axA_lab, axC_lab)
    linkxaxes!(axA_rwa, axC_rwa)
    xlims!(axC_lab, TSTART, LAB_TSTOP)
    xlims!(axC_rwa, TSTART, TSTOP)

    rowgap!(fig.layout, 4)
    rowgap!(fig.layout, 2, 12)   # extra space so amplitude x-ticks clear the legend

    # Save to the DrWatson plots dir, and (only if the Overleaf dir exists) copy
    # PDFs to Figures/ and PNGs to FiguresPNG/.
    overleaf = normpath(projectdir("..", "FilonProjectOverleaf"))
    isdir(overleaf) || @warn "paper repo not found; figure saved to plots/ only" overleaf maxlog=1
    overleaf_subdir = Dict("pdf" => "Figures", "png" => "FiguresPNG")
    for (ext, kw) in (("pdf", (; pt_per_unit = 1)), ("png", (; px_per_unit = 3)))
        fname = "$(basename).$(ext)"
        dests = [plotsdir("cnot3", fname)]
        isdir(overleaf) && push!(dests, joinpath(overleaf, overleaf_subdir[ext], fname))
        for p in dests
            mkpath(dirname(p))
            save(p, fig; kw...)
            println("  saved → ", p)
        end
    end
    return fig, states, meanpop
end

println("CNOT3 controls + state evolution: rotating frame=$ROT_FRAME, lab frame, " *
        "display [$TSTART, $TSTOP] ns, init=|$(INIT_SBA[1])$(INIT_SBA[2])$(INIT_SBA[3])⟩")
fig, states, meanpop = make_figure()
println("Most populous states (mean population over 0–$TFULL ns, $ROT_FRAME frame):")
for i in states
    i0 = i - 1
    println("  |$(i0 ÷ 16)$((i0 ÷ 4) % 4)$(i0 % 4)⟩  meanpop=", round(meanpop[i]; digits = 4))
end
println("Done.")
fig
