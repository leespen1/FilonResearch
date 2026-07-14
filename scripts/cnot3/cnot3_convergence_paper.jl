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
# The companion speedup/Δt tables are produced by cnot3_tables_paper.jl, which
# shares src/cnot3_paper_common.jl but does not load Makie.
#
# Figures are written to the DrWatson plots dir, and (only if the Overleaf dir
# exists) copied into FilonProjectOverleaf: PDFs to Figures/, PNGs to FiguresPNG/.
#
# Run:  julia --project=. scripts/cnot3/cnot3_convergence_paper.jl
#       (CNOT3_INIT=basis to restrict to one initial condition)
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using CairoMakie

include(srcdir("cnot3_paper_common.jl"))   # constants, run loading, frame_df, ...

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

const METHODS = (:Filon, :ControlledFilon, :Hermite)   # display / legend order
const METHOD_LABELS = Dict(:Filon => "Filon", :ControlledFilon => "Controlled Filon",
                           :Hermite => "Hermite")
# Compact variant for narrow legends (e.g. the vertical right-side GMRES legend).
const METHOD_LABELS_SHORT = Dict(:Filon => "Filon", :ControlledFilon => "C-Filon",
                                 :Hermite => "Hermite")
const METHOD_COLOR = Dict(:Filon           => Makie.wong_colors()[1],   # blue
                          :ControlledFilon => Makie.wong_colors()[2],   # orange
                          :Hermite         => Makie.wong_colors()[3])   # green
const METHOD_LS = Dict(:Filon => :solid, :ControlledFilon => (:dash, 1.0),
                       :Hermite => (:dashdot, 1.0))
const ORDER_MARKERS = (:circle, :rect, :diamond)

# Paper sizing: SIAM single column \linewidth = 5.125 in, 1 pt = 1/72 in.
const PAPER_PT_PER_IN = 72
const PAPER_WIDTH_IN  = 5.125

# The method/order legend is included by default; set CNOT3_LEGEND=0 to omit it
# (e.g. when the Rabi convergence figure's identical legend is already in view).
const SHOW_LEGEND = get(ENV, "CNOT3_LEGEND", "1") ∈ ("1", "true", "yes")

# Logarithmic minor ticks (2..9 × 10^k) over a generous exponent range; Makie
# clips them to each axis's visible window.
const LOG_MINOR_TICKS = vec([m * 10.0^k for m in 2:9, k in -8:8])

# Largest "nice" value (1/2/5 × 10^k) at most x — a clean lower axis limit that
# sits just below the data instead of leaving a thin autoscaled sliver.
function nice_floor(x)
    e = floor(Int, log10(x))
    m = x / 10.0^e
    base = m >= 5 ? 5.0 : m >= 2 ? 2.0 : 1.0
    return base * 10.0^e
end

# Method/order legend (+ optional extra group); concrete Vectors, never Vector{Any}
# (a Vector{Any} trips a Makie/ComputePipeline text-rendering bug at save time).
function method_order_legend(legend_extra; order_asymptotic = true,
                             method_labels = METHOD_LABELS)
    method_entries = [LineElement(color = METHOD_COLOR[m], linestyle = METHOD_LS[m], linewidth = 2)
                      for m in METHODS]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[si], color = :black, markersize = 8)
                      for si in 1:length(SVALS)]
    # The GMRES iteration count is not asymptotic in Δt, so its legend drops the
    # O(Δtᵖ) tag and shows the bare s value.
    order_labels = order_asymptotic ?
        [L"s=%$(s)\;(\mathcal{O}(\Delta t^{%$(2(s+1))}))" for s in SVALS] :
        [L"s=%$(s)" for s in SVALS]
    method_labs = [method_labels[m] for m in METHODS]
    legend_extra === nothing &&
        return ([method_entries, order_entries], [method_labs, order_labels], ["Method", "Order"])
    return ([method_entries, order_entries, legend_extra[1]],
            [method_labs, order_labels, legend_extra[2]],
            ["Method", "Order", legend_extra[3]])
end

# Write fig to the DrWatson plots dir and (if it exists) the Overleaf dir:
# PDFs to Figures/, PNGs to FiguresPNG/.
function save_figure(fig, basename)
    overleaf = projectdir("FilonProjectOverleaf")
    overleaf_subdir = Dict("pdf" => "Figures", "png" => "FiguresPNG")
    for (ext, unit) in (("pdf", 1), ("png", 3))
        fname = "$(basename).$(ext)"
        dests = [plotsdir("cnot3", fname)]
        isdir(overleaf) && push!(dests, joinpath(overleaf, overleaf_subdir[ext], fname))
        for p in dests
            mkpath(dirname(p))
            ext == "pdf" ? save(p, fig; pt_per_unit = unit) : save(p, fig; px_per_unit = unit)
            println("  saved → ", p)
        end
    end
end

# -----------------------------------------------------------------------------
# Shared two-panel scaffold: `draw_panel!(ax, df)` draws one frame's curves.
# -----------------------------------------------------------------------------
function paper_2panel(df_lab, df_rwa, draw_panel!; title, xlabel, ylabel,
                      xticks = Makie.automatic, yticks = Makie.automatic,
                      xlims = (nothing, nothing),
                      ylims = (nothing, nothing), ylims_rwa = nothing,
                      link_yaxis = true, yminor = false,
                      legend = true, legend_extra = nothing, legend_position = :bottom,
                      order_asymptotic = true, basename)
    right_legend = legend && legend_position == :right
    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    # A right-side legend sits beside the panels, so no extra bottom row is needed.
    fig = Figure(size = (W, (legend && !right_legend) ? 216 : 178),
                 fontsize = 8, figure_padding = (2, 3, 2, 2))
    title_span = right_legend ? (1:3) : (1:2)
    Label(fig[1, title_span], title; fontsize = 10, font = :bold)

    # The rwa panel may carry its own y-range (only meaningful when y is unshared).
    rwa_ylims = ylims_rwa === nothing ? ylims : ylims_rwa
    yminor_kw = yminor ? (; yminorticksvisible = true, yminorticks = LOG_MINOR_TICKS) : (;)
    ax_lab = Axis(fig[2, 1]; title = "Lab Frame", xlabel = xlabel, ylabel = ylabel,
                  xscale = log10, yscale = log10, xticks = xticks, yticks = yticks,
                  limits = (xlims, ylims), yminor_kw...)
    ax_rwa = Axis(fig[2, 2]; title = "RWA Frame", xlabel = xlabel,
                  xscale = log10, yscale = log10, xticks = xticks, yticks = yticks,
                  limits = (xlims, rwa_ylims), yminor_kw...)
    draw_panel!(ax_lab, df_lab)
    draw_panel!(ax_rwa, df_rwa)
    # Share both axes (linked) or only the x-axis; either way the rwa panel keeps
    # its own y tick labels so each side can be read directly.
    link_yaxis ? linkaxes!(ax_lab, ax_rwa) : linkxaxes!(ax_lab, ax_rwa)

    if right_legend
        groups, labs, titls = method_order_legend(legend_extra; order_asymptotic,
                                                  method_labels = METHOD_LABELS_SHORT)
        Legend(fig[2, 3], groups, labs, titls;
               orientation = :vertical, framevisible = true, titleposition = :top,
               patchsize = (14f0, 8f0), rowgap = 2, titlegap = 3,
               labelsize = 7, titlesize = 7, padding = (4f0, 4f0, 3f0, 3f0))
    elseif legend
        groups, labs, titls = method_order_legend(legend_extra; order_asymptotic)
        Legend(fig[3, 1:2], groups, labs, titls;
               orientation = :horizontal, framevisible = true, titleposition = :left,
               nbanks = 3, patchsize = (14f0, 8f0), colgap = 5, titlegap = 4,
               labelsize = 7, titlesize = 7, padding = (4f0, 4f0, 3f0, 3f0))
    end
    rowgap!(fig.layout, 1, 1)
    legend && !right_legend && rowgap!(fig.layout, 2, 4)
    colgap!(fig.layout, 1, 6)
    right_legend && colgap!(fig.layout, 2, 6)

    save_figure(fig, basename)
    return fig
end

# -----------------------------------------------------------------------------
# 2x2 scaffold: stack two frame-pair quantities (rows) over a shared legend.
# Each `spec` is a NamedTuple describing one row: (draw!, xlabel, ylabel, xlims,
# ylims, xticks, yticks).  Columns are lab (left) | rwa (right), linked per row.
# -----------------------------------------------------------------------------
function paper_4panel(df_lab, df_rwa, top, bottom; title, legend = true,
                      legend_extra = nothing, basename)
    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    fig = Figure(size = (W, legend ? 373 : 335), fontsize = 8, figure_padding = (2, 3, 2, 2))
    Label(fig[1, 1:2], title; fontsize = 10, font = :bold)

    function frame_row!(row, spec, coltitles; xlabelpadding = 3.0, link_yaxis = true, yminor = false)
        yminor_kw = yminor ? (; yminorticksvisible = true, yminorticks = LOG_MINOR_TICKS) : (;)
        ax_lab = Axis(fig[row, 1]; title = coltitles ? "Lab Frame" : "",
                      xlabel = spec.xlabel, ylabel = spec.ylabel, xlabelpadding = xlabelpadding,
                      xscale = log10, yscale = log10, xticks = spec.xticks, yticks = spec.yticks,
                      limits = (spec.xlims, spec.ylims), yminor_kw...)
        ax_rwa = Axis(fig[row, 2]; title = coltitles ? "RWA Frame" : "",
                      xlabel = spec.xlabel, xlabelpadding = xlabelpadding,
                      xscale = log10, yscale = log10, xticks = spec.xticks, yticks = spec.yticks,
                      limits = (spec.xlims, get(spec, :ylims_rwa, spec.ylims)), yminor_kw...)
        spec.draw!(ax_lab, df_lab)
        spec.draw!(ax_rwa, df_rwa)
        # Either way the rwa panel keeps its own y tick labels (read each side directly).
        link_yaxis ? linkaxes!(ax_lab, ax_rwa) : linkxaxes!(ax_lab, ax_rwa)
    end
    # Pull the top row's x-label tight against its tick labels so it reads as that
    # row's axis label, not a heading for the bottom row.  The work-precision row
    # (bottom) keeps each frame's own y-axis since their elapsed-time ranges differ.
    frame_row!(2, top, true; xlabelpadding = 0.0)
    frame_row!(3, bottom, false; link_yaxis = false, yminor = true)

    if legend
        groups, labs, titls = method_order_legend(legend_extra)
        Legend(fig[4, 1:2], groups, labs, titls;
               orientation = :horizontal, framevisible = true, titleposition = :left,
               nbanks = 3, patchsize = (14f0, 8f0), colgap = 5, titlegap = 4,
               labelsize = 7, titlesize = 7, padding = (4f0, 4f0, 3f0, 3f0))
    end
    rowgap!(fig.layout, 1, 1)
    rowgap!(fig.layout, 2, 8)
    legend && rowgap!(fig.layout, 3, 4)
    colgap!(fig.layout, 1, 6)
    # Make the convergence (top) row a touch shorter than the work-precision row.
    rowsize!(fig.layout, 2, Auto(0.9))
    rowsize!(fig.layout, 3, Auto(1.0))

    save_figure(fig, basename)
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

# -----------------------------------------------------------------------------
# The three figures for one initial condition.
# -----------------------------------------------------------------------------
function figures_for(df_all, init)
    df_lab = frame_df(df_all, "lab", init)
    df_rwa = frame_df(df_all, "rwa", init)
    rwa_error = norm(uref_of("rwa", init) .- uref_of("norwa", init))
    icl = ic_title_suffix(init)
    println("  init=$init: RWA modeling error = ", round(rwa_error; sigdigits = 4))

    vof(col) = sub -> Vector{Float64}(sub[!, col])
    dtof = sub -> Vector{Float64}(sub.dt)

    # 1. Convergence: Δt vs final-time error (y capped at 1e0).
    # Major ticks on every decade; Makie clips the x ticks to the visible range.
    ypows = -8:0
    xpows = -8:0
    # Clamp the lower Δt limit to a clean value just below the finest run, so the
    # left edge does not sit a sliver short of a decade.
    dt_min = min(minimum(df_lab.dt), minimum(df_rwa.dt))
    conv_xlims = (nice_floor(dt_min), 1.5)
    conv!(ax, df) = begin
        hlines!(ax, rwa_error; color = :firebrick, linestyle = :dot, linewidth = 1.5)
        draw_series!(ax, df, dtof, vof(:final_error); keepfn = (d, e) -> in_window(e))
    end
    paper_2panel(df_lab, df_rwa, conv!;
        title = "CNOT Gate Convergence$icl", xlabel = L"\Delta t",
        ylabel = "Final-Time Error", xlims = conv_xlims, ylims = (1e-8, 1e0),
        xticks = (10.0 .^ xpows, [L"10^{%$p}" for p in xpows]),
        yticks = (10.0 .^ ypows, [L"10^{%$p}" for p in ypows]),
        legend = SHOW_LEGEND, legend_extra = FIREBRICK_LEGEND,
        basename = "cnot3_convergence_labrwa_$(init)")

    # 2. Work-precision: final-time error vs elapsed time (lines in nsteps order).
    # The lab and rwa elapsed-time ranges differ, so each frame keeps its own
    # y-axis, with limits chosen per IC; only the Final-Time Error x-axis is shared.
    wp_ylims_lab, wp_ylims_rwa = init == "basis"   ? ((1e2, 1e4), (1e0, 1e3)) :
                                 init == "uniform"  ? ((1e1, 1e4), (10.0^-0.5, 1e3)) :
                                 ((nothing, nothing), (nothing, nothing))
    # Decade major ticks so the 2..9 minor ticks subdivide them cleanly.
    wp_ypows = -2:5
    wp_yticks = (10.0 .^ wp_ypows, [L"10^{%$p}" for p in wp_ypows])
    wp!(ax, df) = begin
        vlines!(ax, rwa_error; color = :firebrick, linestyle = :dot, linewidth = 1.5)
        draw_series!(ax, df, vof(:final_error), vof(:t_elapsed); keepfn = (e, t) -> in_window(e))
    end
    paper_2panel(df_lab, df_rwa, wp!;
        title = "CNOT Work–Precision$icl", xlabel = "Final-Time Error",
        ylabel = "Time to Compute (s)", xlims = (1e-8, 1e0), yticks = wp_yticks,
        ylims = wp_ylims_lab, ylims_rwa = wp_ylims_rwa, link_yaxis = false, yminor = true,
        legend = SHOW_LEGEND, legend_extra = FIREBRICK_LEGEND,
        basename = "cnot3_workprecision_labrwa_$(init)")

    # 2b. Combined: convergence (top) over work-precision (bottom), one shared legend.
    decade_ticks = (10.0 .^ xpows, [L"10^{%$p}" for p in xpows])
    conv_spec = (draw! = conv!, xlabel = L"\Delta t", ylabel = "Final-Time Error",
                 xlims = conv_xlims, ylims = (1e-8, 1e0),
                 xticks = decade_ticks, yticks = decade_ticks)
    wp_spec = (draw! = wp!, xlabel = "Final-Time Error", ylabel = "Time to Compute (s)",
               xlims = (1e-8, 1e0), ylims = wp_ylims_lab, ylims_rwa = wp_ylims_rwa,
               xticks = decade_ticks, yticks = wp_yticks)
    paper_4panel(df_lab, df_rwa, conv_spec, wp_spec;
        title = "CNOT Convergence & Work–Precision$icl",
        legend = SHOW_LEGEND, legend_extra = FIREBRICK_LEGEND,
        basename = "cnot3_convergence_workprecision_labrwa_$(init)")

    # 3. GMRES iterations: Δt vs mean GMRES iterations per step.
    gm!(ax, df) = draw_series!(ax, df, dtof, sub -> Vector{Float64}(coalesce.(sub.gmres_mean, NaN)))
    paper_2panel(df_lab, df_rwa, gm!;
        title = "CNOT GMRES Iterations$icl", xlabel = L"\Delta t",
        ylabel = "Mean GMRES Iterations / Step", legend = SHOW_LEGEND,
        legend_position = :right, order_asymptotic = false,
        basename = "cnot3_gmres_labrwa_$(init)")
end

df_all = load_cnot3_runs()
for init in INITS
    figures_for(df_all, init)
end
println("Done.")
