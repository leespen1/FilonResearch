# =============================================================================
# CNOT3 paper figures (lab + RWA frames), SIAM-single-column styling
# =============================================================================
#
# Produces three two-panel (lab | RWA) figures, in one shared style, for each
# initial condition (essential gate basis and uniform superposition):
#
#   1. convergence   — Δt (x) vs final-time error (y)
#   2. work-precision — final-time error (x) vs elapsed solve time (y); points
#                       are connected in order of increasing nsteps
#   3. gmres          — Δt (x) vs mean GMRES iterations per step (y)
#
# Method = colour + linestyle, order s = marker.  "Hermite" is the
# controlled-Hermite method (the ω = 0 Filon), which runs on the same
# ControlledOperator / GMRES path as the Filon methods, so all three curves are
# compared like-for-like.  Errors are measured against the cached Vern9 reference
# for each (frame, initialCondition) — precompute with cnot3_collect_reference.jl.
#
# Figures are written only to the DrWatson plots dir (NOT the Overleaf dir).
#
# Run:  julia --project=. scripts/cnot3/cnot3_convergence_paper.jl
#       (CNOT3_INIT=basis to restrict to one initial condition)
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using DataFrames
using LinearAlgebra
using CairoMakie

# Lightweight: reads jld2 only (no ODE solver); references must already exist.
include(srcdir("error_analysis.jl"))   # load_vern9_reference

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

const prefix   = get(ENV, "CNOT3_PREFIX", "cnot3Convergence")
const datapath = datadir(prefix)
# Both initial conditions by default; CNOT3_INIT restricts to one.
const INITS = haskey(ENV, "CNOT3_INIT") ? [ENV["CNOT3_INIT"]] : ["basis", "uniform"]

# Problem size (must match the collected sweep).
const NOSC = 10
const NGUARD = 2
const TMAX = 550.0
const NSAVES = 16

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

ic_label(ic) = ic == "basis" ? "gate basis IC" :
               ic == "uniform" ? "uniform IC" : "$(ic) IC"

uref_of(frame, init) = load_vern9_reference(; frame, initialCondition = init, Nosc = NOSC,
                                            Nguard = NGUARD, Tmax = TMAX, nsaves = NSAVES)["uref"]

# Per-(frame, init) DataFrame with the derived columns the figures need.
function frame_df(df_all, frame, init)
    mask = isequal.(df_all.initialCondition, init) .& isequal.(df_all.frame, frame)
    df = copy(df_all[mask, :])
    isempty(df) && error("No runs for frame=$frame, init=$init in $datapath")
    uref = uref_of(frame, init)
    df.dt = TMAX ./ df.nsteps
    df.final_error = [norm(h[:, end] .- uref) for h in df.history]
    return df
end

# Rows for one (method, s), sorted by increasing nsteps.
seriesof(df, m, s) = sort(df[(df.method .== m) .& (df.s .== s), :], :nsteps)

# -----------------------------------------------------------------------------
# Shared two-panel scaffold: `draw_panel!(ax, df)` draws one frame's curves.
# -----------------------------------------------------------------------------
function paper_2panel(df_lab, df_rwa, draw_panel!; title, xlabel, ylabel,
                      yticks = Makie.automatic, xlims = (nothing, nothing),
                      ylims = (nothing, nothing), legend_extra = nothing, basename)
    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    fig = Figure(size = (W, 240), fontsize = 8, figure_padding = (2, 3, 2, 2))
    Label(fig[1, 1:2], title; fontsize = 10, font = :bold)

    ax_lab = Axis(fig[2, 1]; title = "Lab Frame", xlabel = xlabel, ylabel = ylabel,
                  xscale = log10, yscale = log10, yticks = yticks, limits = (xlims, ylims))
    ax_rwa = Axis(fig[2, 2]; title = "RWA Frame", xlabel = xlabel,
                  xscale = log10, yscale = log10, yticks = yticks, limits = (xlims, ylims))
    draw_panel!(ax_lab, df_lab)
    draw_panel!(ax_rwa, df_rwa)
    linkyaxes!(ax_lab, ax_rwa)
    hideydecorations!(ax_rwa, grid = false)

    method_entries = [LineElement(color = METHOD_COLOR[m], linestyle = METHOD_LS[m], linewidth = 2)
                      for m in METHODS]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[si], color = :black, markersize = 8)
                      for si in 1:length(SVALS)]
    order_labels = [L"s=%$(s)\;(\mathcal{O}(\Delta t^{%$(2(s+1))}))" for s in SVALS]
    # Build the legend argument vectors concretely (NOT Vector{Any}): a Vector{Any}
    # here trips a Makie/ComputePipeline text-rendering bug at save time.
    if legend_extra === nothing
        groups = [method_entries, order_entries]
        labs   = [[METHOD_LABELS[m] for m in METHODS], order_labels]
        titls  = ["Method", "Order"]
    else
        groups = [method_entries, order_entries, legend_extra[1]]
        labs   = [[METHOD_LABELS[m] for m in METHODS], order_labels, legend_extra[2]]
        titls  = ["Method", "Order", legend_extra[3]]
    end
    Legend(fig[3, 1:2], groups, labs, titls;
           orientation = :horizontal, framevisible = true, titleposition = :left,
           nbanks = 3, patchsize = (14f0, 8f0), colgap = 5, titlegap = 4,
           labelsize = 7, titlesize = 7, padding = (4f0, 4f0, 3f0, 3f0))
    rowgap!(fig.layout, 1, 1)
    rowgap!(fig.layout, 2, 4)

    for (ext, unit) in (("pdf", 1), ("png", 3))
        p = plotsdir("cnot3", "$(basename).$(ext)")
        mkpath(dirname(p))
        ext == "pdf" ? save(p, fig; pt_per_unit = unit) : save(p, fig; px_per_unit = unit)
        println("  saved → ", p)
    end
    return fig
end

# Scatter-line one (method, s) series with x/y vectors already chosen.
function draw_series!(ax, df, xof, yof; keepfn = (x, y) -> trues(length(x)))
    for m in METHODS, (si, s) in enumerate(SVALS)
        sub = seriesof(df, m, s)
        isempty(sub) && continue
        x = xof(sub); y = yof(sub)
        keep = isfinite.(x) .& isfinite.(y) .& keepfn(x, y)
        any(keep) || continue
        scatterlines!(ax, x[keep], y[keep]; color = METHOD_COLOR[m], linestyle = METHOD_LS[m],
                      marker = ORDER_MARKERS[si], markersize = 7, linewidth = 1.3)
    end
end

const FIREBRICK_LEGEND = ([LineElement(color = :firebrick, linestyle = :dot, linewidth = 2)],
                          ["RWA Error"], "Modeling")
in_window(e) = (e .>= ERROR_WINDOW[1]) .& (e .<= ERROR_WINDOW[2])

# -----------------------------------------------------------------------------
# The three figures for one initial condition.
# -----------------------------------------------------------------------------
function figures_for(df_all, init)
    df_lab = frame_df(df_all, "lab", init)
    df_rwa = frame_df(df_all, "rwa", init)
    rwa_error = norm(uref_of("rwa", init) .- uref_of("norwa", init))
    icl = ic_label(init)
    println("  init=$init: RWA modeling error = ", round(rwa_error; sigdigits = 4))

    vof(col) = sub -> Vector{Float64}(sub[!, col])
    dtof = sub -> Vector{Float64}(sub.dt)

    # 1. Convergence: Δt vs final-time error (y capped at 1e0).
    ypows = -8:2:0
    conv!(ax, df) = begin
        hlines!(ax, rwa_error; color = :firebrick, linestyle = :dot, linewidth = 1.5)
        draw_series!(ax, df, dtof, vof(:final_error); keepfn = (d, e) -> in_window(e))
    end
    paper_2panel(df_lab, df_rwa, conv!;
        title = "CNOT3 Gate Convergence ($icl)", xlabel = L"\Delta t",
        ylabel = "Final Time Error", xlims = (nothing, 1.5), ylims = (1e-8, 1e0),
        yticks = (10.0 .^ ypows, [L"10^{%$p}" for p in ypows]),
        legend_extra = FIREBRICK_LEGEND, basename = "cnot3_convergence_labrwa_$(init)")

    # 2. Work-precision: final-time error vs elapsed time (lines in nsteps order).
    wp!(ax, df) = begin
        vlines!(ax, rwa_error; color = :firebrick, linestyle = :dot, linewidth = 1.5)
        draw_series!(ax, df, vof(:final_error), vof(:t_elapsed); keepfn = (e, t) -> in_window(e))
    end
    paper_2panel(df_lab, df_rwa, wp!;
        title = "CNOT3 Work–Precision ($icl)", xlabel = "Final Time Error",
        ylabel = "Elapsed Time (s)", xlims = (1e-8, 1e0),
        legend_extra = FIREBRICK_LEGEND, basename = "cnot3_workprecision_labrwa_$(init)")

    # 3. GMRES iterations: Δt vs mean GMRES iterations per step.
    gm!(ax, df) = draw_series!(ax, df, dtof, sub -> Vector{Float64}(coalesce.(sub.gmres_mean, NaN)))
    paper_2panel(df_lab, df_rwa, gm!;
        title = "CNOT3 GMRES Iterations ($icl)", xlabel = L"\Delta t",
        ylabel = "Mean GMRES Iterations / Step", basename = "cnot3_gmres_labrwa_$(init)")
end

println("Reading ", datapath)
df_all = collect_results(datapath)
df_all.method = Symbol.(df_all.method)
for init in INITS
    figures_for(df_all, init)
end
println("Done.")
