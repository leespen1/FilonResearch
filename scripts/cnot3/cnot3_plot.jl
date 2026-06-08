# =============================================================================
# CNOT3 convergence — plotting stage
# =============================================================================
#
# Reads the per-run result files written by `cnot3_convergence_collect_data.jl`,
# gathers them into a DataFrame with `collect_results`, computes each run's error
# against the *finest run* (treated as the reference / ground truth), then plots
# a rabi-style log-log convergence figure (and a wall-clock-time figure).
#
# Data collection and plotting are deliberately separate: everything below works
# off the DataFrame, so filtering ("ignore errors above X", "only these nsteps
# for this method", …) is just a DataFrame operation — see `FILTERS` below.
#
# Run:  julia --project=. \
#         scripts/cnot3/cnot3_plot.jl
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using DataFrames
using LinearAlgebra
using Printf
using CairoMakie

include(srcdir("error_analysis.jl"))   # l2_integral_error_subsample

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())
const inch = 96

const prefix = get(ENV, "CNOT3_PREFIX", "cnot3Convergence")
# Plot the current commit's data by default; override with CNOT3_COMMIT to plot
# a specific past collection (the subdirectory name under datadir(prefix)).
const commit = get(ENV, "CNOT3_COMMIT", gitdescribe(projectdir()))
const datapath = datadir(prefix, commit)
# Each initial condition has its own row count and its own finest-run reference,
# so we plot one at a time.  Override with CNOT3_INIT (e.g. "uniform").
const init = get(ENV, "CNOT3_INIT", "basis")

# Display order / labels / styling for the methods.
const METHOD_ORDER  = (:hermite, :filon, :controlled_filon)
const METHOD_LABELS = Dict(
    :hermite          => "Hermite (QGD)",
    :filon            => "Filon",
    :controlled_filon => "Controlled-Filon",
)
const SVALS         = (0, 1, 2)
const METHOD_COLORS = Dict(m => c for (m, c) in zip(METHOD_ORDER, Makie.wong_colors()))
const ORDER_MARKERS = (:circle, :rect, :diamond)

# -----------------------------------------------------------------------------
# Filters (edit these to slice the data without recollecting)
# -----------------------------------------------------------------------------
# Errors outside this window are dropped before plotting: the upper bound hides
# diverged coarse-step runs, the lower bound hides the round-off floor.
const ERROR_WINDOW = (1e-13, 1e1)
# Optionally restrict the timestep range, globally or per method.  `nothing`
# means "no restriction".
const NSTEPS_WINDOW = Dict{Symbol,Any}()   # e.g. :filon => (2^4, 2^14)

# -----------------------------------------------------------------------------
# Load + derive errors
# -----------------------------------------------------------------------------

df = collect_results(datapath)
isempty(df) && error("No result files found in $(datapath). Run the collection script first.")
df = df[df.initialCondition .== init, :]
isempty(df) && error("No runs with initialCondition=$(init) in $(datapath). " *
                     "Set CNOT3_INIT to one that was collected.")
df.method = Symbol.(df.method)
println("initialCondition = $(init)")
println("Loaded $(nrow(df)) runs: ",
        join(["$(METHOD_LABELS[m])×$(count(==(m), df.method))" for m in unique(df.method)], ", "))

# Finest-run reference: the most-refined solution overall (largest nsteps, ties
# broken by the highest order s).  Its final-time state is the ground truth.
sort!(df, [:nsteps, :s], rev = true)
ref = first(df)
Tmax = ref.Tmax
uref = ref.history[:, end]
href = ref.history
@printf("Reference (finest run): method=%s  s=%d  nsteps=%d\n",
        METHOD_LABELS[ref.method], ref.s, ref.nsteps)

df.final_error = [norm(h[:, end] .- uref) for h in df.history]
df.l2_error    = [l2_integral_error_subsample(h, href, Tmax) for h in df.history]

# Which error column to plot.
const ERROR_COL = :final_error

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

"Rows for one (method, s), filtered and sorted by nsteps, as (nsteps, error)."
function curve(df, method, s; error_col = ERROR_COL)
    sub = df[(df.method .== method) .& (df.s .== s), :]
    win = get(NSTEPS_WINDOW, method, nothing)
    if win !== nothing
        sub = sub[(sub.nsteps .>= win[1]) .& (sub.nsteps .<= win[2]), :]
    end
    err = sub[!, error_col]
    keep = (err .>= ERROR_WINDOW[1]) .& (err .<= ERROR_WINDOW[2]) .& isfinite.(err)
    sub = sub[keep, :]
    sort!(sub, :nsteps)
    return Vector{Float64}(sub.nsteps), Vector{Float64}(sub[!, error_col])
end

"""
Combined log-log convergence panel: colour = method, marker = order s, with
dotted-grey O(Δtᵖ) guide lines anchored to the first available curve of each
order.  Saves png/svg/pdf to `plotsdir("cnot3")`.
"""
function make_convergence_figure(df, methods; basename = "cnot3_convergence")
    fig = Figure(size = (7.5inch, 4.6inch), fontsize = 11)
    Label(fig[0, 1:2],
          L"\textrm{CNOT3\;gate\;convergence}\;\;(N_{\mathrm{osc}}=%$(ref.nOscLevels),\;T=%$(Tmax))";
          fontsize = 13, padding = (0, 0, 6, 0))
    ax = Axis(fig[1, 1];
              xlabel = "Number of timesteps", ylabel = "Final-time 2-norm error",
              xscale = log10, yscale = log10)

    # Global nsteps span (for guide lines).
    all_n = Float64[]
    for m in methods, s in SVALS
        n, _ = curve(df, m, s); append!(all_n, n)
    end
    isempty(all_n) && error("Nothing to plot after filtering.")
    nfine = exp10.(range(log10(minimum(all_n)), log10(maximum(all_n)), length = 200))

    # O(Δtᵖ) guide, anchored to the first method/order with data.
    for (si, s) in enumerate(SVALS)
        p = 2 * (s + 1)
        for m in methods
            n, e = curve(df, m, s)
            isempty(n) && continue
            C = e[1] * n[1]^p
            lines!(ax, nfine, C ./ (nfine .^ p);
                   color = :gray, linestyle = :dot, linewidth = 1.5)
            break
        end
    end

    for m in methods, (si, s) in enumerate(SVALS)
        n, e = curve(df, m, s)
        isempty(n) && continue
        scatterlines!(ax, n, e; color = METHOD_COLORS[m], marker = ORDER_MARKERS[si],
                      markersize = 9, linewidth = 1.5)
    end

    method_entries = [LineElement(color = METHOD_COLORS[m], linewidth = 2) for m in methods]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[j], color = :black, markersize = 9)
                      for j in 1:length(SVALS)]
    ref_entry      = [LineElement(color = :gray, linestyle = :dot, linewidth = 1.5)]
    Legend(fig[1, 2],
           [method_entries, order_entries, ref_entry],
           [[METHOD_LABELS[m] for m in methods],
            [L"s=0\;(O(\Delta t^2))", L"s=1\;(O(\Delta t^4))", L"s=2\;(O(\Delta t^6))"],
            [L"O(\Delta t^p)\;\textrm{guide}"]],
           ["Method", "Order", "Slope"],
           orientation = :vertical, tellheight = false, tellwidth = true)
    colsize!(fig.layout, 1, Relative(0.72))

    mkpath(plotsdir("cnot3"))
    for ext in ("png", "svg", "pdf")
        save(plotsdir("cnot3", "$(basename).$(ext)"), fig)
    end
    println("  saved → ", plotsdir("cnot3", "$(basename).{png,svg,pdf}"))
    return fig
end

"Wall-clock time vs nsteps (log-log), same colour/marker scheme."
function make_timing_figure(df, methods; basename = "cnot3_timing")
    fig = Figure(size = (7.5inch, 4.6inch), fontsize = 11)
    Label(fig[0, 1:2], L"\textrm{CNOT3\;wall-clock\;time}"; fontsize = 13, padding = (0, 0, 6, 0))
    ax = Axis(fig[1, 1]; xlabel = "Number of timesteps", ylabel = "Solve time (s)",
              xscale = log10, yscale = log10)
    for m in methods, (si, s) in enumerate(SVALS)
        sub = df[(df.method .== m) .& (df.s .== s), :]
        isempty(sub) && continue
        sort!(sub, :nsteps)
        scatterlines!(ax, Vector{Float64}(sub.nsteps), Vector{Float64}(sub.t_elapsed);
                      color = METHOD_COLORS[m], marker = ORDER_MARKERS[si],
                      markersize = 9, linewidth = 1.5)
    end
    method_entries = [LineElement(color = METHOD_COLORS[m], linewidth = 2) for m in methods]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[j], color = :black, markersize = 9)
                      for j in 1:length(SVALS)]
    Legend(fig[1, 2], [method_entries, order_entries],
           [[METHOD_LABELS[m] for m in methods], ["s=0", "s=1", "s=2"]],
           ["Method", "Order"]; orientation = :vertical, tellheight = false, tellwidth = true)
    colsize!(fig.layout, 1, Relative(0.72))
    mkpath(plotsdir("cnot3"))
    for ext in ("png", "svg", "pdf")
        save(plotsdir("cnot3", "$(basename).$(ext)"), fig)
    end
    println("  saved → ", plotsdir("cnot3", "$(basename).{png,svg,pdf}"))
    return fig
end

methods_present = [m for m in METHOD_ORDER if m in df.method]
fig_conv = make_convergence_figure(df, methods_present)
fig_time = make_timing_figure(df, methods_present)
println("Done.")
fig_conv
