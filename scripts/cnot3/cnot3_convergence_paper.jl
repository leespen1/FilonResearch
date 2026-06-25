# =============================================================================
# CNOT3 convergence — paper figure (lab + RWA frames), Rabi-figure styling
# =============================================================================
#
# A combined two-panel convergence figure (lab and RWA frames, sharing a y-axis)
# in the same style as scripts/rabi_oscillator/rabi_frames_convergence.jl: paper
# sized (SIAM single column), method = colour + linestyle, order s = marker, Δt
# on the x-axis, a horizontal three-section legend below, saved as a vector PDF
# (plus a PNG preview) to both the DrWatson plots dir and the Overleaf Figures dir.
#
# Reads the per-run data produced by cnot3_convergence_collect_data.jl (one frame
# at a time, each with its own finest-run reference) and shows three methods:
# Hermite (QGD), Filon, Controlled Filon.  The exploratory per-frame multi-figure
# script (cnot3_plot.jl) is left untouched.
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

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

const prefix   = get(ENV, "CNOT3_PREFIX", "cnot3Convergence")
const commit   = get(ENV, "CNOT3_COMMIT", gitdescribe(projectdir()))
const datapath = datadir(prefix, commit)
const init     = get(ENV, "CNOT3_INIT", "basis")

# Method = colour + linestyle (so coincident curves stay distinct), order = marker.
const METHODS = (:filon, :controlled_filon, :hermite)        # display / legend order
const METHOD_LABELS = Dict(:filon => "Filon", :controlled_filon => "Controlled Filon",
                           :hermite => "Hermite (QGD)")
const METHOD_COLOR = Dict(:filon            => Makie.wong_colors()[1],   # blue
                          :controlled_filon => Makie.wong_colors()[2],   # orange
                          :hermite          => Makie.wong_colors()[3])   # green
const METHOD_LS = Dict(:filon => :solid, :controlled_filon => (:dash, 1.0),
                       :hermite => (:dashdot, 1.0))
const SVALS = (0, 1, 2)
const ORDER_MARKERS = (:circle, :rect, :diamond)

# Drop diverged coarse runs (upper) and the round-off floor (lower) before plotting.
const ERROR_WINDOW = (1e-13, 1e1)

# Paper sizing: SIAM single column \linewidth = 5.125 in, 1 pt = 1/72 in.
const PAPER_PT_PER_IN = 72
const PAPER_WIDTH_IN  = 5.125

# -----------------------------------------------------------------------------
# Data: per-frame DataFrame with a final-time error column vs that frame's
# finest run (largest order, then deepest nsteps).
# -----------------------------------------------------------------------------
function frame_errors(frame)
    df = collect_results(datapath)
    df = df[(df.initialCondition .== init) .& (df.frame .== frame), :]
    isempty(df) && error("No runs for frame=$frame, init=$init in $datapath")
    df.method = Symbol.(df.method)
    sort!(df, [:s, :nsteps], rev = true)             # finest = highest order, deepest
    ref = first(df)
    uref = ref.history[:, end]
    df.final_error = [norm(h[:, end] .- uref) for h in df.history]
    return df, ref.Tmax, ref
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

"Draw one frame's curves on `ax`: O(Δtᵖ) slope guides (grey dotted) plus the
method × order scatter-lines (method = colour+linestyle, order s = marker)."
function plot_frame!(ax, df, Tmax; methods = METHODS)
    # O(Δtᵖ) guide per order: least-squares slope-p fit (error ≈ C·Δtᵖ) over that
    # order's steep, not-yet-floored data, drawn only across that data's Δt range
    # so it hugs the curves.
    for s in SVALS
        p = 2 * (s + 1)
        dts = Float64[]; es = Float64[]
        for m in methods
            d, e = curve(df, Tmax, m, s); append!(dts, d); append!(es, e)
        end
        isempty(dts) && continue
        steep = es .> 3 * minimum(es)
        count(steep) >= 2 || (steep = trues(length(es)))
        dc = dts[steep]; ec = es[steep]
        logC = sum(log.(ec) .- p .* log.(dc)) / length(dc)   # error = C·Δtᵖ
        dspan = exp10.(range(log10(minimum(dc)), log10(maximum(dc)), length = 50))
        lines!(ax, dspan, exp(logC) .* (dspan .^ p);
               color = :gray, linestyle = :dot, linewidth = 1.2)
    end
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
    df_lab, T_lab, ref_lab = frame_errors("lab")
    df_rwa, T_rwa, _        = frame_errors("rwa")

    # y range / even-power ticks from the plotted (filtered) data of both panels.
    allE = Float64[]
    for (df, T) in ((df_lab, T_lab), (df_rwa, T_rwa)), m in METHODS, s in SVALS
        _, e = curve(df, T, m, s); append!(allE, e)
    end
    lo = floor(Int, log10(minimum(allE))); hi = ceil(Int, log10(maximum(allE)))
    iseven(lo) || (lo -= 1)
    ypows = lo:2:hi
    yticks = (10.0 .^ ypows, [L"10^{%$p}" for p in ypows])
    ylims = (10.0^lo / 3, 10.0^hi * 3)

    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    fig = Figure(size = (W, 240), fontsize = 8, figure_padding = (2, 3, 2, 2))
    Label(fig[1, 1:2], "CNOT3 Gate Convergence"; fontsize = 10, font = :bold)

    ax_lab = Axis(fig[2, 1]; title = "Lab Frame", xlabel = L"\Delta t",
                  ylabel = "Final-time 2-norm error", xscale = log10, yscale = log10,
                  yticks = yticks, limits = (nothing, ylims))
    ax_rwa = Axis(fig[2, 2]; title = "RWA Frame", xlabel = L"\Delta t",
                  xscale = log10, yscale = log10, yticks = yticks,
                  limits = (nothing, ylims))
    plot_frame!(ax_lab, df_lab, T_lab)
    plot_frame!(ax_rwa, df_rwa, T_rwa)
    linkyaxes!(ax_lab, ax_rwa)
    hideydecorations!(ax_rwa, grid = false)

    method_entries = [LineElement(color = METHOD_COLOR[m], linestyle = METHOD_LS[m], linewidth = 2)
                      for m in METHODS]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[si], color = :black, markersize = 8)
                      for si in 1:length(SVALS)]
    order_labels = [L"s=%$(s)\;(\mathcal{O}(\Delta t^{%$(2(s+1))}))" for s in SVALS]
    slope_entry = [LineElement(color = :gray, linestyle = :dot, linewidth = 2)]
    Legend(fig[3, 1:2],
           [method_entries, order_entries, slope_entry],
           [[METHOD_LABELS[m] for m in METHODS], order_labels, [L"\mathcal{O}(\Delta t^p)"]],
           ["Method", "Order", "Slope"];
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
