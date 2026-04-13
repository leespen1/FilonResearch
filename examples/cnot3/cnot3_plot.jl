# cnot3_plot.jl
#
# Plot convergence data collected by cnot3_collect_data.jl.
#
# Usage:
#   julia --project=examples examples/cnot3/cnot3_plot.jl path/to/summary.jld2
#   julia --project=examples examples/cnot3/cnot3_plot.jl   # uses most recent summary.jld2 in data/

using JLD2
using CairoMakie
using LaTeXStrings
using LinearAlgebra

# ============================================================
# Load data
# ============================================================

if !isempty(ARGS)
    datafile = ARGS[1]
else
    data_dir = joinpath(@__DIR__, "data")
    # Find summary.jld2 files in experiment subdirectories
    summary_files = String[]
    for d in readdir(data_dir, join=true)
        isdir(d) || continue
        sf = joinpath(d, "summary.jld2")
        isfile(sf) && push!(summary_files, sf)
    end
    isempty(summary_files) && error("No summary.jld2 files found in $data_dir. Run cnot3_collect_data.jl first.")
    datafile = sort(summary_files, by=mtime)[end]
    println("Using most recent data file: $(datafile)")
end

data = load(datafile)

hermite_data    = data["hermite_data"]
filon_zero_data = data["filon_zero_data"]
filon_rot_data  = data["filon_rot_data"]
hermite_times     = data["hermite_times"]
filon_zero_times  = data["filon_zero_times"]
filon_rot_times   = data["filon_rot_times"]
tplot        = data["tplot"]
ctrl_p_vals  = data["ctrl_p_vals"]
ctrl_q_vals  = data["ctrl_q_vals"]
t_fine       = data["t_fine"]
hist_fine    = data["hist_fine"]
s_finest     = data["s_finest"]
ns_fine      = data["ns_fine"]
s_values     = data["s_values"]
min_power    = data["min_power"]
max_power    = data["max_power"]
Tmax         = data["Tmax"]
run_hermite    = data["run_hermite"]
run_filon_zero = data["run_filon_zero"]
plot_prefix    = data["plot_prefix"]

# ============================================================
# Helper
# ============================================================

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

# ============================================================
# Plotting setup
# ============================================================

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

# ---- Figure 1: Control Functions ----

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

# ---- Figure 3: Method Convergence (Richardson Errors) ----

fig3 = Figure(size=(min(6.5, 2.5*n_s) * inch, 3.5inch), fontsize=12)
Label(fig3[0, 1:n_s],
    L"\textbf{Richardson Extrapolation Error Estimates}",
    fontsize=14)

ax_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)
    h_nsteps, h_hists = hermite_data[s]
    f0_nsteps, f0_hists = filon_zero_data[s]
    fr_nsteps, fr_hists = filon_rot_data[s]

    ax = Axis(fig3[1, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Error}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )

    # Hermite
    if run_hermite
        h_re = richardson_errors(h_hists, order, Tmax)
        scatterlines!(ax, h_nsteps[1:end-1], h_re,
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end

    # Filon (w=0)
    if run_filon_zero
        f0_re = richardson_errors(f0_hists, order, Tmax)
        scatterlines!(ax, f0_nsteps[1:end-1], f0_re,
            color=filon0_color, marker=filon0_marker, label=L"\mathrm{Filon}\;(\omega=0)")
    end

    # Filon (rot)
    fr_re = richardson_errors(fr_hists, order, Tmax)
    scatterlines!(ax, fr_nsteps[1:end-1], fr_re,
        color=filonr_color, marker=filonr_marker, label=L"\mathrm{Filon}\;(\omega=\mathrm{rot})")

    # Reference slope
    ref_ns = [2.0^k for k in min_power:max_power]
    ref_C = fr_re[1] * fr_nsteps[1]^order
    ref_line = ref_C ./ (ref_ns .^ order)
    lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")

    if col == 1
        ax_first = ax
    end
end
Legend(fig3[2, 1:n_s], ax_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_convergence_richardson_$(plot_prefix).png"), fig3)
println("Saved: Plots/cnot3_convergence_richardson_$(plot_prefix).png")

# ---- Figure 3b: Wall-clock time per method ----

fig3b = Figure(size=(min(6.5, 2.5*n_s) * inch, 3.5inch), fontsize=12)
Label(fig3b[0, 1:n_s],
    L"\textbf{Wall-Clock Time per Method}",
    fontsize=14)

ax_time_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)
    h_nsteps, _ = hermite_data[s]
    f0_nsteps, _ = filon_zero_data[s]
    fr_nsteps, _ = filon_rot_data[s]

    ax = Axis(fig3b[1, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Time\; (s)}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )

    if run_hermite
        scatterlines!(ax, h_nsteps, hermite_times[s],
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end

    if run_filon_zero
        scatterlines!(ax, f0_nsteps, filon_zero_times[s],
            color=filon0_color, marker=filon0_marker, label=L"\mathrm{Filon}\;(\omega=0)")
    end

    scatterlines!(ax, fr_nsteps, filon_rot_times[s],
        color=filonr_color, marker=filonr_marker, label=L"\mathrm{Filon}\;(\omega=\mathrm{rot})")

    # Linear reference (ideal O(nsteps) scaling)
    ref_ns = [2.0^k for k in min_power:max_power]
    ref_base = run_hermite ? hermite_times[s][1] : filon_rot_times[s][1]
    ref_line = ref_base .* (ref_ns ./ ref_ns[1])
    lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label=L"O(\mathrm{nsteps})")

    if col == 1
        ax_time_first = ax
    end
end
Legend(fig3b[2, 1:n_s], ax_time_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_timing_$(plot_prefix).png"), fig3b)
println("Saved: Plots/cnot3_timing_$(plot_prefix).png")

# ---- Figure 4: Combined (controls, state evolution, convergence) ----

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

    fr_re = richardson_errors(fr_hists, order, Tmax)
    scatterlines!(ax, fr_nsteps[1:end-1], fr_re,
        color=filonr_color, marker=filonr_marker, label="Filon (w=rot)")

    ref_ns = [2.0^k for k in min_power:max_power]
    ref_C = fr_re[1] * fr_nsteps[1]^order
    ref_line = ref_C ./ (ref_ns .^ order)
    lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")

    if col == 1
        ax4_first = ax
    end
end
Legend(fig4[5, 1:n_s], ax4_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "cnot3_combined_$(plot_prefix).png"), fig4)
println("Saved: Plots/cnot3_combined_$(plot_prefix).png")

display(fig1)
display(fig2)
display(fig3)
display(fig3b)
display(fig4)
