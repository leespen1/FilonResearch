# =============================================================================
# CNOT3 convergence — paper figure (lab + RWA frames), Rabi-figure styling
# =============================================================================
#
# A combined two-panel convergence figure (lab and RWA frames, sharing a y-axis)
# in the same style as scripts/rabi_oscillator/rabi_frames_convergence.jl: paper
# sized (SIAM single column), method = colour + linestyle, order s = marker, Δt
# on the x-axis, a horizontal legend below, saved as a vector PDF (plus a PNG
# preview) to both the DrWatson plots dir and the Overleaf Figures dir.
#
# Reference: like the Rabi figure, each frame's error is measured against a Vern9
# (adaptive RK) solution of that frame's own ODE at abstol=reltol=1e-13 — an
# independent reference, not the finest Hermite run.  The RWA modeling error (the
# firebrick ceiling) is the Vern9 RWA-frame vs Vern9 no-RWA-frame final-state
# difference: the two solve the same rotating-frame dynamics differing only by the
# dropped counter-rotating terms, so their separation at T is the accuracy floor
# no RWA-frame computation can beat.  The Vern9 references are cached.
#
# Run (data is commit-namespaced; point at the collection commit):
#   CNOT3_COMMIT=46407571d844c93a725b139e763f21dfa1bc1fcc \
#     julia --project=. scripts/cnot3/cnot3_convergence_paper.jl
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using DataFrames
using LinearAlgebra
using CairoMakie
using FilonResearch
using QuantumGateDesign
using OrdinaryDiffEqVerner

# Problem builders + make_initial_condition + qgd_to_controlled_operator.
include(srcdir("cnot3_run.jl"))

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

const prefix   = get(ENV, "CNOT3_PREFIX", "cnot3Convergence")
const commit   = get(ENV, "CNOT3_COMMIT", gitdescribe(projectdir()))
const datapath = datadir(prefix, commit)
const init     = get(ENV, "CNOT3_INIT", "basis")

# Problem size (must match the collected sweep).
const NOSC = 10
const NGUARD = 2
const TMAX = 550.0

# Method = colour + linestyle (so coincident curves stay distinct), order = marker.
# "Hermite" here is the efficient controlled-Hermite method (the ω=0 Filon), which
# is numerically identical to the regular Hermite method but runs on the same
# ControlledOperator path as the Filon methods.
const METHODS = (:filon, :controlled_filon, :controlled_hermite)   # display / legend order
const METHOD_LABELS = Dict(:filon => "Filon", :controlled_filon => "Controlled Filon",
                           :controlled_hermite => "Hermite")
const METHOD_COLOR = Dict(:filon              => Makie.wong_colors()[1],   # blue
                          :controlled_filon   => Makie.wong_colors()[2],   # orange
                          :controlled_hermite => Makie.wong_colors()[3])   # green
const METHOD_LS = Dict(:filon => :solid, :controlled_filon => (:dash, 1.0),
                       :controlled_hermite => (:dashdot, 1.0))
const SVALS = (0, 1, 2)
const ORDER_MARKERS = (:circle, :rect, :diamond)

# Drop diverged coarse runs (upper) and the round-off floor (lower) before plotting.
const ERROR_WINDOW = (1e-13, 1e1)

# Paper sizing: SIAM single column \linewidth = 5.125 in, 1 pt = 1/72 in.
const PAPER_PT_PER_IN = 72
const PAPER_WIDTH_IN  = 5.125

# -----------------------------------------------------------------------------
# Vern9 reference: solve dψ/dt = A(t)ψ on a frame's own ODE at 1e-13, one column
# per essential gate state, returning the stacked final state (matches the
# collected history[:, end] layout).  Cached per frame in the campaign data dir.
# -----------------------------------------------------------------------------
function compute_vern9_reference(frame; abstol = 1e-13, reltol = 1e-13)
    fr = Symbol(frame)
    qgd_prob = cnot3_hoho_qgd_prob(N_osc_levels = NOSC, N_guard_levels = NGUARD,
                                   Tmax = TMAX, frame = fr)
    controls, pcof = cnot3_hoho_controls_and_pcof(frame = fr)
    co = qgd_to_controlled_operator(qgd_prob, controls, pcof)   # A(t) = Σ cₖ(t) Aₖ
    ic = make_initial_condition("basis", qgd_prob)              # N_tot × N_ess
    op = Operator(co, 0.0)                                      # reusable A(t) buffer
    function rhs!(du, u, p, t)
        evaluate!(op, co, t)                                   # refresh A(t) in place
        mul!(du, op, u)
        return nothing
    end
    finals = map(eachcol(ic)) do c
        u0 = ComplexF64.(Vector(c))
        sol = solve(ODEProblem(rhs!, u0, (0.0, TMAX)), Vern9();
                    abstol = abstol, reltol = reltol, save_everystep = false)
        Vector{ComplexF64}(sol.u[end])
    end
    return reduce(vcat, finals)
end

function vern9_reference(frame)
    cfg = Dict("frame" => string(frame), "abstol" => 1e-13, "reltol" => 1e-13,
               "Nosc" => NOSC, "Nguard" => NGUARD, "Tmax" => TMAX)
    # Cache in a SEPARATE data dir (not under the run prefix) so collect_results
    # over the run data never picks these reference files up.
    data, _ = produce_or_load(cfg, datadir("cnot3_vern9ref", commit);
                              prefix = "cnot3_vern9ref", tag = false) do _
        println("  computing Vern9 reference for frame=", frame, " (this is the slow step)")
        @strdict uref = compute_vern9_reference(frame)
    end
    return data["uref"]
end

# -----------------------------------------------------------------------------
# Data: per-frame DataFrame with final-time error vs the frame's Vern9 reference.
# -----------------------------------------------------------------------------
function frame_errors(df_all, frame, uref)
    # isequal (not .==) so any non-run rows with `missing` columns filter out cleanly.
    mask = isequal.(df_all.initialCondition, init) .& isequal.(df_all.frame, string(frame))
    df = df_all[mask, :]
    isempty(df) && error("No runs for frame=$frame, init=$init in $datapath")
    Tmax = first(df).Tmax
    df = copy(df)
    df.final_error = [norm(h[:, end] .- uref) for h in df.history]
    return df, Tmax
end

"(Δt, error) for one (method, s): ERROR_WINDOW-filtered, sorted by nsteps."
function curve(df, Tmax, m, s)
    sub = df[(df.method .== m) .& (df.s .== s), :]
    e = sub.final_error
    keep = (e .>= ERROR_WINDOW[1]) .& (e .<= ERROR_WINDOW[2]) .& isfinite.(e)
    sub = sub[keep, :]
    sort!(sub, :nsteps)
    return Tmax ./ Vector{Float64}(sub.nsteps), Vector{Float64}(sub.final_error)
end

"Draw one frame's curves on `ax`: the RWA modeling-error ceiling (firebrick),
then method × order scatter-lines."
function plot_frame!(ax, df, Tmax, rwa_error; methods = METHODS)
    rwa_error === nothing ||
        hlines!(ax, rwa_error; color = :firebrick, linestyle = :dot, linewidth = 1.5)
    for m in methods, (si, s) in enumerate(SVALS)
        d, e = curve(df, Tmax, m, s)
        isempty(d) && continue
        scatterlines!(ax, d, e; color = METHOD_COLOR[m], linestyle = METHOD_LS[m],
                      marker = ORDER_MARKERS[si], markersize = 7, linewidth = 1.3)
    end
    return ax
end

# -----------------------------------------------------------------------------
# Combined figure
# -----------------------------------------------------------------------------
function make_combined_figure(; basename = "cnot3_convergence_labrwa")
    # Independent Vern9 references (cached); RWA modeling error = RWA vs no-RWA.
    uref_lab   = vern9_reference("lab")
    uref_rwa   = vern9_reference("rwa")
    uref_norwa = vern9_reference("norwa")
    rwa_error  = norm(uref_rwa .- uref_norwa)
    println("RWA modeling error (Vern9 RWA vs no-RWA at T): ", round(rwa_error; sigdigits = 4))

    # Exclude norwa: the figure shows only lab + RWA, and skipping norwa avoids
    # mmap-ing any norwa result file that a concurrent collection job is mid-write.
    df_all = collect_results(datapath; rexclude = [r"frame=norwa"])
    df_all.method = Symbol.(df_all.method)
    df_lab, T_lab = frame_errors(df_all, "lab", uref_lab)
    df_rwa, T_rwa = frame_errors(df_all, "rwa", uref_rwa)

    # Axis window: crop the coarse-Δt blowup band on the right (just past Δt = 1)
    # and floor the y-axis at 1e-8.
    XMAX = 1.5
    YMIN = 1e-8
    # y top from the data visible within the x-window (even power above it).
    visE = Float64[]
    for (df, T) in ((df_lab, T_lab), (df_rwa, T_rwa)), m in METHODS, s in SVALS
        d, e = curve(df, T, m, s)
        for i in eachindex(d)
            d[i] <= XMAX && push!(visE, e[i])
        end
    end
    hi = ceil(Int, log10(maximum(visE)))
    ypows = -8:2:hi
    yticks = (10.0 .^ ypows, [L"10^{%$p}" for p in ypows])
    ylims = (YMIN, 10.0^hi)

    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    fig = Figure(size = (W, 240), fontsize = 8, figure_padding = (2, 3, 2, 2))
    Label(fig[1, 1:2], "CNOT3 Gate Convergence"; fontsize = 10, font = :bold)

    ax_lab = Axis(fig[2, 1]; title = "Lab Frame", xlabel = L"\Delta t",
                  ylabel = "Final Time Error", xscale = log10, yscale = log10,
                  yticks = yticks, limits = ((nothing, XMAX), ylims))
    ax_rwa = Axis(fig[2, 2]; title = "RWA Frame", xlabel = L"\Delta t",
                  xscale = log10, yscale = log10, yticks = yticks,
                  limits = ((nothing, XMAX), ylims))
    plot_frame!(ax_lab, df_lab, T_lab, rwa_error)
    plot_frame!(ax_rwa, df_rwa, T_rwa, rwa_error)
    linkyaxes!(ax_lab, ax_rwa)
    hideydecorations!(ax_rwa, grid = false)

    method_entries = [LineElement(color = METHOD_COLOR[m], linestyle = METHOD_LS[m], linewidth = 2)
                      for m in METHODS]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[si], color = :black, markersize = 8)
                      for si in 1:length(SVALS)]
    order_labels = [L"s=%$(s)\;(\mathcal{O}(\Delta t^{%$(2(s+1))}))" for s in SVALS]
    modeling_entries = [LineElement(color = :firebrick, linestyle = :dot, linewidth = 2)]
    Legend(fig[3, 1:2],
           [method_entries, order_entries, modeling_entries],
           [[METHOD_LABELS[m] for m in METHODS], order_labels, ["RWA Error"]],
           ["Method", "Order", "Modeling"];
           orientation = :horizontal, framevisible = true, titleposition = :left,
           nbanks = 3, patchsize = (14f0, 8f0), colgap = 5, titlegap = 4,
           labelsize = 7, titlesize = 7, padding = (4f0, 4f0, 3f0, 3f0))
    rowgap!(fig.layout, 1, 1)
    rowgap!(fig.layout, 2, 4)

    pdf_paths = [plotsdir("cnot3", "$(basename).pdf"),
                 projectdir("FilonProjectOverleaf", "Figures", "$(basename).pdf")]
    png_paths = [plotsdir("cnot3", "$(basename).png"),
                 projectdir("FilonProjectOverleaf", "FiguresPNG", "$(basename).png")]
    for p in pdf_paths
        mkpath(dirname(p)); save(p, fig; pt_per_unit = 1); println("  saved → ", p)
    end
    for p in png_paths
        mkpath(dirname(p)); save(p, fig; px_per_unit = 3); println("  saved → ", p)
    end
    return fig
end

println("Reading ", datapath)
make_combined_figure()
println("Done.")
