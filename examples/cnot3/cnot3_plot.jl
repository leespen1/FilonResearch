# cnot3_plot.jl
#
# Plot convergence data collected by cnot3_collect_data.jl.
#
# Scans the per-run JLD2 files (method=*_s=*_nsteps=*.jld2) in an experiment
# directory and builds the summary on the fly, so different methods/orders
# may have different amounts of data.
#
# Usage:
#   julia --project=examples examples/cnot3/cnot3_plot.jl path/to/experiment_dir
#   julia --project=examples examples/cnot3/cnot3_plot.jl   # uses most recent experiment dir in data/

t0 = time()

using JLD2
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using QuantumGateDesign



# ============================================================
# Locate experiment directory
# ============================================================
@info "Locate experiment directory" t=time()-t0

if !isempty(ARGS)
    experiment_dir = ARGS[1]
    # Backwards compat: if a .jld2 file is passed, use its parent directory
    if isfile(experiment_dir) && endswith(experiment_dir, ".jld2")
        experiment_dir = dirname(experiment_dir)
    end
else
    data_dir = joinpath(@__DIR__, "data")
    # Find subdirectories containing at least one method=*.jld2 file
    exp_dirs = String[]
    for d in readdir(data_dir, join=true)
        isdir(d) || continue
        any(f -> startswith(f, "method=") && endswith(f, ".jld2"), readdir(d)) || continue
        push!(exp_dirs, d)
    end
    isempty(exp_dirs) && error("No experiment directories found in $data_dir. Run cnot3_collect_data.jl first.")
    experiment_dir = sort(exp_dirs, by=mtime)[end]
    println("Using most recent experiment directory: $(experiment_dir)")
end

# ============================================================
# Load metadata (stored in every per-run jld2 file)
# ============================================================
@info "Load metadata" t=time()-t0

const FNAME_RE = r"^method=([A-Za-z0-9_]+)_s=(\d+)_nsteps=(\d+)\.jld2$"

_run_files = filter(f -> match(FNAME_RE, f) !== nothing, readdir(experiment_dir))
isempty(_run_files) && error("No per-run method=*.jld2 files in $experiment_dir. Run cnot3_collect_data.jl first.")
metadata = load(joinpath(experiment_dir, first(_run_files)))
Tmax        = metadata["Tmax"]
plot_prefix = metadata["plot_prefix"]
Cfreq       = metadata["Cfreq"]
pcof        = metadata["pcof"]
degree      = metadata["degree"]
D1          = metadata["D1"]

# ============================================================
# Scan per-run files and build summary
# ============================================================
@info "Scan per-run files and build summary" t=time()-t0

# method => (s => (nsteps_vec, hists_vec))
method_hists = Dict{String, Dict{Int, Tuple{Vector{Int}, Vector{Matrix{ComplexF64}}}}}()
method_times = Dict{String, Dict{Int, Vector{Float64}}}()

for fname in readdir(experiment_dir)
    m = match(FNAME_RE, fname)
    m === nothing && continue

    method = String(m[1])
    s      = parse(Int, m[2])
    nsteps = parse(Int, m[3])

    fdata = load(joinpath(experiment_dir, fname))

    msub = get!(method_hists, method, Dict{Int, Tuple{Vector{Int}, Vector{Matrix{ComplexF64}}}}())
    tsub = get!(method_times, method, Dict{Int, Vector{Float64}}())
    if !haskey(msub, s)
        msub[s] = (Int[], Matrix{ComplexF64}[])
        tsub[s] = Float64[]
    end
    push!(msub[s][1], nsteps)
    push!(msub[s][2], fdata["hist"])
    push!(tsub[s], fdata["elapsed"])
end

# Sort each (method, s) series by nsteps
for (method, sdict) in method_hists
    for (s, (ns_vec, hist_vec)) in sdict
        perm = sortperm(ns_vec)
        sdict[s] = (ns_vec[perm], hist_vec[perm])
        method_times[method][s] = method_times[method][s][perm]
    end
end

# Expose per-method dicts with the same names the rest of the script used
empty_hists() = Dict{Int, Tuple{Vector{Int}, Vector{Matrix{ComplexF64}}}}()
empty_times() = Dict{Int, Vector{Float64}}()

hermite_data    = get(method_hists, "hermite",    empty_hists())
filon_zero_data = get(method_hists, "filon_zero", empty_hists())
filon_rot_data  = get(method_hists, "filon_rot",  empty_hists())
hermite_times    = get(method_times, "hermite",    empty_times())
filon_zero_times = get(method_times, "filon_zero", empty_times())
filon_rot_times  = get(method_times, "filon_rot",  empty_times())

run_hermite    = !isempty(hermite_data)
run_filon_zero = !isempty(filon_zero_data)
run_filon_rot  = !isempty(filon_rot_data)

# Union of s values across all methods
s_values = sort(collect(union(keys(hermite_data), keys(filon_zero_data), keys(filon_rot_data))))
isempty(s_values) && error("No per-run data found in $experiment_dir.")

# nsteps range (for reference slope lines)
all_nsteps = Int[]
for sdict in values(method_hists), (ns_vec, _) in values(sdict)
    append!(all_nsteps, ns_vec)
end
min_power = Int(round(log2(minimum(all_nsteps))))
max_power = Int(round(log2(maximum(all_nsteps))))

println("Methods found: ", join(sort(collect(keys(method_hists))), ", "))
println("s values: ", s_values)
for method in sort(collect(keys(method_hists)))
    for s in sort(collect(keys(method_hists[method])))
        ns_vec, _ = method_hists[method][s]
        println("  $method  s=$s  nsteps=", ns_vec)
    end
end

# ============================================================
# Helper
# ============================================================
@info "Helper" t=time()-t0

# L2-in-time error (needed for Richardson error computation)
function l2_time_error_subsample(hist_fine, hist_coarse, T)
    N_fine = size(hist_fine, 2) - 1
    N_coarse = size(hist_coarse, 2) - 1
    stride = N_fine ÷ N_coarse
    dt = T / N_coarse
    err_sq = sum(norm(hist_fine[:, 1 + (k-1)*stride] - hist_coarse[:, k])^2 for k in 1:N_coarse+1)
    return sqrt(dt * err_sq)
end

function richardson_errors(hists, order, T)
    return [l2_time_error_subsample(hists[i+1], hists[i], T) * 2^order / (2^order - 1)
            for i in 1:length(hists)-1]
end

# Replace any interior outlier (point > `threshold` × geometric mean of its
# immediate neighbors) with the geometric mean of those neighbors — i.e.,
# nearest-neighbor interpolation in log-log space. Returns a modified copy.
function sanitize_outliers(errs; threshold=3.0)
    out = copy(errs)
    for i in 2:length(out)-1
        gm = sqrt(errs[i-1] * errs[i+1])
        if isfinite(gm) && gm > 0 && errs[i] > threshold * gm
            out[i] = gm
        end
    end
    return out
end

# Log-log least-squares fit err(n) = C * n^b, then invert for each target error
# to estimate how many timesteps are needed. Returns NaN when the fit is
# degenerate; Inf when the slope is non-negative (no convergence observed).
function nsteps_for_target(ns_vec, errs, targets)
    mask = [isfinite(e) && e > 0 for e in errs]
    xs = log10.(Float64.(collect(ns_vec)[mask]))
    ys = log10.(Float64.(collect(errs)[mask]))
    length(xs) >= 2 || return fill(NaN, length(targets))
    x̄ = sum(xs) / length(xs); ȳ = sum(ys) / length(ys)
    sxx = sum((xs .- x̄).^2)
    sxx > 0 || return fill(NaN, length(targets))
    b = sum((xs .- x̄) .* (ys .- ȳ)) / sxx
    a = ȳ - b * x̄
    b < 0 || return fill(Inf, length(targets))
    return [10.0^((log10(t) - a) / b) for t in targets]
end

function true_solution_errors(nsteps_vec, hist_vec, nsteps_true, hist_true, T)
    errs = Float64[]
    for (ns, h) in zip(nsteps_vec, hist_vec)
        if h === hist_true
            push!(errs, NaN)
        elseif ns <= nsteps_true && nsteps_true % ns == 0
            # "true" is finer (or equal) — subsample true at candidate's grid
            push!(errs, l2_time_error_subsample(hist_true, h, T))
        elseif ns >  nsteps_true && ns % nsteps_true == 0
            # candidate is finer than truth — subsample candidate at truth's grid.
            # Resulting "error" floors at the true solution's own accuracy.
            push!(errs, l2_time_error_subsample(h, hist_true, T))
        else
            push!(errs, NaN)
        end
    end
    return errs
end

function get_controls(method_order::Integer, D1::Integer, Cfreq::AbstractMatrix{<: Real}, tf::Real)
    degree = method_order
    base_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
    base_control = QuantumGateDesign.FortranBSplineControl2(base_bspline, tf)
    controls = [CarrierControl(base_control, freqs) for freqs in eachrow(Cfreq)]
    return controls
end

# ============================================================
# Plotting setup
# ============================================================
@info "Plotting setup" t=time()-t0

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())
inch = 96
colors = Makie.wong_colors()

hermite_color = colors[1]
filon0_color  = colors[2]
filonr_color  = colors[3]

hermite_marker = :circle
filon0_marker  = :rect
filonr_marker  = :diamond

mkpath(joinpath(@__DIR__, "../..", "Plots"))

ctrl_labels = [L"p_1(t)", L"p_2(t)", L"p_3(t)", L"q_1(t)", L"q_2(t)", L"q_3(t)"]
n_s = length(s_values)

# Fixed axis limits for all error plots (adjust as needed)
const ERROR_YLIMS = (1e-6, 2.5e1)
const ERROR_XLIMS = (1e1, nothing)

# Shared log-decade ticks from 1e-15 to 1e15 (most fall outside axis limits).
const TICK_EXPONENTS = -15:15
const TICK_VALUES    = 10.0 .^ TICK_EXPONENTS
const TICK_LABELS    = [L"10^{%$e}" for e in TICK_EXPONENTS]
const ERROR_TICKS    = (TICK_VALUES, TICK_LABELS)
apply_error_ticks!(ax) = (ax.xticks = ERROR_TICKS; ax.yticks = ERROR_TICKS)

# Combined-plot styling: color encodes s (Wong), linestyle encodes method
# family (Hermite dashed, Filon solid). Markers stay per-method for redundancy.
const METHOD_ORDER = ("hermite", "filon_zero", "filon_rot")
const METHOD_STYLE = Dict(
    "hermite"    => (marker=:circle,  linestyle=:dash),
    "filon_zero" => (marker=:rect,    linestyle=:solid),
    "filon_rot"  => (marker=:diamond, linestyle=:solid),
)

function combined_label(key::AbstractString, s::Integer)
    if key == "hermite"
        return L"\mathrm{Hermite},\; s=%$s"
    elseif key == "filon_zero"
        return L"\mathrm{Filon}(\omega=0),\; s=%$s"
    else
        return L"\mathrm{Filon},\; s=%$s"
    end
end

# ---- Pick "true" reference solution: run with smallest Richardson error ----
@info "Pick 'true' reference solution: run with smallest Richardson error" t=time()-t0

best_re = Inf
best_method = ""
best_s = -1
best_nsteps_true = -1
best_hist_true = nothing
for (method, sdict) in method_hists
    for (s, (ns_vec, hist_vec)) in sdict
        length(hist_vec) >= 2 || continue
        order = 2*(s+1)
        errs = richardson_errors(hist_vec, order, Tmax)
        for i in eachindex(errs)
            if errs[i] < best_re
                global best_re = errs[i]
                global best_method = method
                global best_s = s
                global best_nsteps_true = ns_vec[i+1]
                global best_hist_true = hist_vec[i+1]
            end
        end
    end
end
best_hist_true === nothing && error("Need at least one (method, s) with ≥ 2 runs to pick a reference solution.")
println("\n\"True\" solution: method=$best_method, s=$best_s, nsteps=$best_nsteps_true  (Richardson err ≈ $best_re)")

# ---- Figure 1: Control Functions ----
@info "Figure 1: Control Functions" t=time()-t0

const Tmax_original = 550.0
rot_controls = get_controls(degree, D1, Cfreq, Tmax_original)
nplot = 1000
ctrl_p_vals = Matrix{Float64}(undef, length(rot_controls), nplot)
ctrl_q_vals = Matrix{Float64}(undef, length(rot_controls), nplot)
tplot = collect(range(0, Tmax, length=nplot))
for (i, ctrl) in enumerate(rot_controls)
    pcof_i = QuantumGateDesign.get_control_vector_slice(pcof, rot_controls, i)
    for j in 1:nplot
        ctrl_p_vals[i, j] = eval_p_derivative(ctrl, tplot[j], pcof_i, 0)
        ctrl_q_vals[i, j] = eval_q_derivative(ctrl, tplot[j], pcof_i, 0)
    end
end

fig1 = Figure(size=(6.5inch, 4inch), fontsize=12)
ax1 = Axis(fig1[1, 1],
    xlabel=L"t",
    ylabel=L"\mathrm{Control\; amplitude}",
    title=L"\textbf{Control Functions}",
)

Nctrl = size(ctrl_p_vals, 1)
for i in 1:Nctrl
    lines!(ax1, tplot, ctrl_p_vals[i, :], color=colors[i], label=ctrl_labels[i])
    lines!(ax1, tplot, ctrl_q_vals[i, :], color=colors[i+3], linestyle=:dash, label=ctrl_labels[i+3])
end
Legend(fig1[1, 2], ax1, framevisible=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_control_functions_$(plot_prefix).png"), fig1)
println("\nSaved: Plots/cnot3_control_functions_$(plot_prefix).png")

# ---- Figure 2: State Evolution ----
@info "Figure 2: State Evolution" t=time()-t0
#=

N = size(hist_fine, 1)

fig2 = Figure(size=(6.5inch, 5.5inch), fontsize=12)
ax2_re = Axis(fig2[1, 1],
    ylabel=L"\mathrm{Re}(u_k)",
    title=L"\textbf{State Evolution}\;\;(\mathrm{nsteps}=%$ns_fine,\; s=%$s_finest)",
)
ax2_im = Axis(fig2[2, 1],
    xlabel=L"t",
    ylabel=L"\mathrm{Im}(u_k)",
)

state_cmap = cgrad(:viridis, N, categorical=true)
for k in 1:N
    lines!(ax2_re, t_fine, real.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
    lines!(ax2_im, t_fine, imag.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
end
Colorbar(fig2[1:2, 2], colormap=:viridis, limits=(1, N), label=L"k")

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_state_evolution_$(plot_prefix).png"), fig2)
println("Saved: Plots/cnot3_state_evolution_$(plot_prefix).png")
=#

# ---- Figure 3: Method Convergence (Richardson Errors) ----
@info "Figure 3: Method Convergence (Richardson Errors)" t=time()-t0

fig3 = Figure(size=(min(6.5, 2.5*n_s) * inch, 3.5inch), fontsize=12)
Label(fig3[0, 1:n_s],
    L"\textbf{Richardson Extrapolation Error Estimates}",
    #L"\textbf{Error in State Vector}",
    fontsize=14)

ax_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)

    ax = Axis(fig3[1, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Error}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )
    ylims!(ax, ERROR_YLIMS...)
    xlims!(ax, ERROR_XLIMS...)
    apply_error_ticks!(ax)

    # Hermite
    if haskey(hermite_data, s)
        h_nsteps, h_hists = hermite_data[s]
        if length(h_hists) >= 2
            h_re = richardson_errors(h_hists, order, Tmax)
            if s == 0
                h_re = sanitize_outliers(h_re)
            end
            scatterlines!(ax, h_nsteps[1:end-1], h_re,
                color=hermite_color, marker=hermite_marker, label="Hermite")
        end
    end

    # Filon (w=0)
    if haskey(filon_zero_data, s)
        f0_nsteps, f0_hists = filon_zero_data[s]
        if length(f0_hists) >= 2
            f0_re = richardson_errors(f0_hists, order, Tmax)
            scatterlines!(ax, f0_nsteps[1:end-1], f0_re,
                color=filon0_color, marker=filon0_marker, label=L"\mathrm{Filon}\;(\omega=0)")
        end
    end

    # Filon (rot)
    if haskey(filon_rot_data, s)
        fr_nsteps, fr_hists = filon_rot_data[s]
        if length(fr_hists) >= 2
            fr_re = richardson_errors(fr_hists, order, Tmax)
            scatterlines!(ax, fr_nsteps[1:end-1], fr_re,
                color=filonr_color, marker=filonr_marker, label=L"\mathrm{Filon}\;(\omega=\mathrm{rot})")
        end
    end

    # Reference slope: pick whichever method has ≥2 runs at this s
    ref_re, ref_nsteps = nothing, nothing
    for d in (filon_rot_data, filon_zero_data, hermite_data)
        haskey(d, s) || continue
        nsvec, hvec = d[s]
        length(hvec) >= 2 || continue
        ref_re = richardson_errors(hvec, order, Tmax)
        ref_nsteps = nsvec
        break
    end
    if ref_re !== nothing
        ref_ns = [2.0^k for k in min_power:max_power]
        ref_C = ref_re[1] * ref_nsteps[1]^order
        ref_line = ref_C ./ (ref_ns .^ order)
        lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")
    end

    if col == 1
        global ax_first = ax
    end
end
Legend(fig3[2, 1:n_s], ax_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_convergence_richardson_$(plot_prefix).png"), fig3)
println("Saved: Plots/cnot3_convergence_richardson_$(plot_prefix).png")

# ---- Figure 3b: Wall-clock time per method ----
@info "Figure 3b: Wall-clock time per method" t=time()-t0

fig3b = Figure(size=(min(6.5, 2.5*n_s) * inch, 3.5inch), fontsize=12)
Label(fig3b[0, 1:n_s],
    L"\textbf{Wall-Clock Time per Method}",
    fontsize=14)

ax_time_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)

    ax = Axis(fig3b[1, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Time\; (s)}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )

    if haskey(hermite_data, s)
        h_nsteps, _ = hermite_data[s]
        scatterlines!(ax, h_nsteps, hermite_times[s],
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end

    if haskey(filon_zero_data, s)
        f0_nsteps, _ = filon_zero_data[s]
        scatterlines!(ax, f0_nsteps, filon_zero_times[s],
            color=filon0_color, marker=filon0_marker, label=L"\mathrm{Filon}\;(\omega=0)")
    end

    if haskey(filon_rot_data, s)
        fr_nsteps, _ = filon_rot_data[s]
        scatterlines!(ax, fr_nsteps, filon_rot_times[s],
            color=filonr_color, marker=filonr_marker, label=L"\mathrm{Filon}\;(\omega=\mathrm{rot})")
    end

    # Linear reference (ideal O(nsteps) scaling)
    ref_base, ref_first_ns = nothing, nothing
    for (dkey, tdict) in (("hermite", hermite_times), ("filon_rot", filon_rot_times), ("filon_zero", filon_zero_times))
        haskey(tdict, s) && !isempty(tdict[s]) || continue
        ref_base = tdict[s][1]
        ref_first_ns = method_hists[dkey][s][1][1]
        break
    end
    if ref_base !== nothing
        ref_ns = [2.0^k for k in min_power:max_power]
        ref_line = ref_base .* (ref_ns ./ ref_first_ns)
        lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label=L"O(\mathrm{nsteps})")
    end

    if col == 1
        global ax_time_first = ax
    end
end
Legend(fig3b[2, 1:n_s], ax_time_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_timing_$(plot_prefix).png"), fig3b)
println("Saved: Plots/cnot3_timing_$(plot_prefix).png")

# ---- Figure 3c: Richardson errors, all s values on one axes ----
@info "Figure 3c: Richardson errors, all s values on one axes" t=time()-t0

fig3c = Figure(size=(7.5inch, 4inch), fontsize=12)
Label(fig3c[0, 1],
    #L"\textbf{Richardson Extrapolation Error Estimates}",
    L"\textbf{Error in State Vector vs. Number of Timesteps}",
    fontsize=14)

ax3c = Axis(fig3c[1, 1],
    xlabel="Number of Timesteps",
    ylabel="Error in State Vector",
    xscale=log10, yscale=log10,
)
ylims!(ax3c, ERROR_YLIMS...)
xlims!(ax3c, ERROR_XLIMS...)
apply_error_ticks!(ax3c)

let wong = Makie.wong_colors()
    for (si, s) in enumerate(s_values)
        order = 2*(s+1)
        s_color = wong[mod1(si, length(wong))]
        for (key, data) in (("hermite", hermite_data), ("filon_zero", filon_zero_data), ("filon_rot", filon_rot_data))
            haskey(data, s) || continue
            ns_vec, hist_vec = data[s]
            length(hist_vec) >= 2 || continue
            errs = richardson_errors(hist_vec, order, Tmax)
            if key == "hermite" && s == 0
                errs = sanitize_outliers(errs)
            end
            style = METHOD_STYLE[key]
            scatterlines!(ax3c, ns_vec[1:end-1], errs,
                color=s_color,
                marker=style.marker, linestyle=style.linestyle,
                label=combined_label(key, s))
        end
    end
end
Legend(fig3c[1, 2], ax3c, framevisible=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_convergence_richardson_combined_$(plot_prefix).png"), fig3c)
save(joinpath(@__DIR__, "../..", "Plots", "cnot3_convergence_richardson_combined_$(plot_prefix).svg"), fig3c)
println("Saved: Plots/cnot3_convergence_richardson_combined_$(plot_prefix).png")

# Extrapolated nsteps required to hit each target Richardson error,
# based on a log-log least-squares fit over the available Richardson data.
const TARGET_ERRORS = [1e-1, 1e-2, 1e-3, 1e-4]
fmt_nsteps(n) = !isfinite(n) ? "   —" :
                n >= 1e12  ? string(round(n/1e12, digits=2), "e12") :
                n >= 1e9   ? string(round(n/1e9,  digits=2), "e9") :
                n >= 1e6   ? string(round(n/1e6,  digits=2), "e6") :
                             string(round(Int, n))
println("\nEstimated nsteps to reach target Richardson error (log-log fit):")
header = rpad("  method / s", 22)
for t in TARGET_ERRORS
    global header *= lpad("err=$t", 14)
end
println(header)
for s in s_values
    for (key, data) in (("hermite", hermite_data), ("filon_zero", filon_zero_data), ("filon_rot", filon_rot_data))
        haskey(data, s) || continue
        ns_vec, hist_vec = data[s]
        length(hist_vec) >= 2 || continue
        order = 2*(s+1)
        errs = richardson_errors(hist_vec, order, Tmax)
        if key == "hermite" && s == 0
            errs = sanitize_outliers(errs)
        end
        nsteps_est = nsteps_for_target(ns_vec[1:end-1], errs, TARGET_ERRORS)
        label = key == "hermite"    ? "Hermite s=$s"    :
                key == "filon_zero" ? "Filon(ω=0) s=$s" :
                                      "Filon(ω=rot) s=$s"
        line = rpad("  " * label, 22)
        for n in nsteps_est
            line *= lpad(fmt_nsteps(n), 14)
        end
        println(line)
    end
end

# ---- Figure 4: True-solution L2 error, per-s panels ----
@info "Figure 4: True-solution L2 error, per-s panels" t=time()-t0

fig4 = Figure(size=(min(6.5, 2.5*n_s) * inch, 3.5inch), fontsize=12)
Label(fig4[0, 1:n_s],
    L"\textbf{Error vs. ``True'' Solution (method=%$best_method,\; s=%$best_s,\; \mathrm{nsteps}=%$best_nsteps_true)}",
    fontsize=14)

ax4_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)

    ax = Axis(fig4[1, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Error}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )
    ylims!(ax, ERROR_YLIMS...)
    xlims!(ax, ERROR_XLIMS...)
    apply_error_ticks!(ax)

    if haskey(hermite_data, s)
        h_nsteps, h_hists = hermite_data[s]
        h_errs = true_solution_errors(h_nsteps, h_hists, best_nsteps_true, best_hist_true, Tmax)
        scatterlines!(ax, h_nsteps, h_errs,
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end
    if haskey(filon_zero_data, s)
        f0_nsteps, f0_hists = filon_zero_data[s]
        f0_errs = true_solution_errors(f0_nsteps, f0_hists, best_nsteps_true, best_hist_true, Tmax)
        scatterlines!(ax, f0_nsteps, f0_errs,
            color=filon0_color, marker=filon0_marker, label=L"\mathrm{Filon}\;(\omega=0)")
    end
    if haskey(filon_rot_data, s)
        fr_nsteps, fr_hists = filon_rot_data[s]
        fr_errs = true_solution_errors(fr_nsteps, fr_hists, best_nsteps_true, best_hist_true, Tmax)
        scatterlines!(ax, fr_nsteps, fr_errs,
            color=filonr_color, marker=filonr_marker, label=L"\mathrm{Filon}")
    end

    # Reference slope: use first non-NaN, non-zero error at this s
    ref_errs, ref_nsteps = nothing, nothing
    for d in (filon_rot_data, filon_zero_data, hermite_data)
        haskey(d, s) || continue
        nsvec, hvec = d[s]
        errs = true_solution_errors(nsvec, hvec, best_nsteps_true, best_hist_true, Tmax)
        idx = findfirst(e -> isfinite(e) && e > 0, errs)
        idx === nothing && continue
        ref_errs = errs
        ref_nsteps = nsvec
        break
    end
    if ref_errs !== nothing
        idx = findfirst(e -> isfinite(e) && e > 0, ref_errs)
        ref_ns = [2.0^k for k in min_power:max_power]
        ref_C = ref_errs[idx] * ref_nsteps[idx]^order
        ref_line = ref_C ./ (ref_ns .^ order)
        lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")
    end

    if col == 1
        global ax4_first = ax
    end
end
Legend(fig4[2, 1:n_s], ax4_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_convergence_true_$(plot_prefix).png"), fig4)
println("Saved: Plots/cnot3_convergence_true_$(plot_prefix).png")

# ---- Figure 4c: True-solution L2 error, all s values on one axes ----
@info "Figure 4c: True-solution L2 error, all s values on one axes" t=time()-t0

fig4c = Figure(size=(7.5inch, 4inch), fontsize=12)
Label(fig4c[0, 1],
    L"\textbf{Error vs. True Solution}",
    fontsize=14)

ax4c = Axis(fig4c[1, 1],
    xlabel=L"\mathrm{nsteps}",
    ylabel=L"\mathrm{Error}",
    xscale=log10, yscale=log10,
)
ylims!(ax4c, ERROR_YLIMS...)
xlims!(ax4c, ERROR_XLIMS...)
apply_error_ticks!(ax4c)

let wong = Makie.wong_colors()
    for (si, s) in enumerate(s_values)
        s_color = wong[mod1(si, length(wong))]
        for (key, data) in (("hermite", hermite_data), ("filon_zero", filon_zero_data), ("filon_rot", filon_rot_data))
            haskey(data, s) || continue
            ns_vec, hist_vec = data[s]
            errs = true_solution_errors(ns_vec, hist_vec, best_nsteps_true, best_hist_true, Tmax)
            style = METHOD_STYLE[key]
            scatterlines!(ax4c, ns_vec, errs,
                color=s_color,
                marker=style.marker, linestyle=style.linestyle,
                label=combined_label(key, s))
        end
    end
end
Legend(fig4c[1, 2], ax4c, framevisible=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_convergence_true_combined_$(plot_prefix).png"), fig4c)
println("Saved: Plots/cnot3_convergence_true_combined_$(plot_prefix).png")

# ---- Figure 4: Combined (controls, state evolution, convergence) ----
@info "Figure 4: Combined (controls, state evolution, convergence)" t=time()-t0
#=

fig4 = Figure(size=(6.5inch, 10inch), fontsize=12)

# Row 1: Control functions
ax4_ctrl = Axis(fig4[1, 1:n_s],
    xlabel=L"t",
    ylabel=L"\mathrm{Control\; amplitude}",
    title=L"\textbf{Control Functions}",
)
for i in 1:Nctrl
    lines!(ax4_ctrl, tplot, ctrl_p_vals[i, :], color=colors[i], label=ctrl_labels[i])
    lines!(ax4_ctrl, tplot, ctrl_q_vals[i, :], color=colors[i+3], linestyle=:dash, label=ctrl_labels[i+3])
end
Legend(fig4[1, n_s+1], ax4_ctrl, framevisible=false)

# Rows 2-3: State evolution (Re and Im)
ax4_re = Axis(fig4[2, 1:n_s],
    ylabel=L"\mathrm{Re}(u_k)",
    title=L"\textbf{State Evolution}\;\;(\mathrm{nsteps}=%$ns_fine,\; s=%$s_finest)",
)
ax4_im = Axis(fig4[3, 1:n_s],
    xlabel=L"t",
    ylabel=L"\mathrm{Im}(u_k)",
)
for k in 1:N
    lines!(ax4_re, t_fine, real.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
    lines!(ax4_im, t_fine, imag.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
end
Colorbar(fig4[2:3, n_s+1], colormap=:viridis, limits=(1, N), label=L"k")

# Row 4: Convergence (Richardson errors)
ax4_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)
    h_nsteps, h_hists = hermite_data[s]
    f0_nsteps, f0_hists = filon_zero_data[s]
    fr_nsteps, fr_hists = filon_rot_data[s]

    ax = Axis(fig4[4, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Error}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )

    if run_hermite
        h_re = richardson_errors(h_hists, order, Tmax)
        scatterlines!(ax, h_nsteps[1:end-1], h_re,
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end

    if run_filon_zero
        f0_re = richardson_errors(f0_hists, order, Tmax)
        scatterlines!(ax, f0_nsteps[1:end-1], f0_re,
            color=filon0_color, marker=filon0_marker, label="Filon (w=0)")
    end

    if run_filon_rot
        fr_re = richardson_errors(fr_hists, order, Tmax)
        scatterlines!(ax, fr_nsteps[1:end-1], fr_re,
            color=filonr_color, marker=filonr_marker, label="Filon (w=rot)")
    end

    ref_ns = [2.0^k for k in min_power:max_power]
    if run_filon_rot
        ref_re = richardson_errors(fr_hists, order, Tmax)
        ref_nsteps_fig4 = fr_nsteps
    elseif run_filon_zero
        ref_re = richardson_errors(f0_hists, order, Tmax)
        ref_nsteps_fig4 = f0_nsteps
    else
        ref_re = richardson_errors(h_hists, order, Tmax)
        ref_nsteps_fig4 = h_nsteps
    end
    ref_C = ref_re[1] * ref_nsteps_fig4[1]^order
    ref_line = ref_C ./ (ref_ns .^ order)
    lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")

    if col == 1
        global ax4_first = ax
    end
end
Legend(fig4[5, 1:n_s], ax4_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_combined_$(plot_prefix).png"), fig4)
println("Saved: Plots/cnot3_combined_$(plot_prefix).png")
=#
