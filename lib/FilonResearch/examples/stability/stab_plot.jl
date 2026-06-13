# Plotting stage for the Filon stability study: reads data/stability/stab_data.jls
# (written by stab_compute.jl) and renders figures to plots/stability/.
# Run under the umbrella env (has CairoMakie), e.g. via srun:
#   srun -p general-short --constraint=intel18 -n1 -c1 --mem=4G -t 0:15:00 \
#       julia --project=. lib/FilonResearch/examples/stability/stab_plot.jl
using Serialization, CairoMakie, Printf

CairoMakie.activate!()
const INCH = 96

const REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const DATA = joinpath(REPO, "data", "stability", "stab_data.jls")
const PLOTS = joinpath(REPO, "plots", "stability"); mkpath(PLOTS)
D = deserialize(DATA)
A, B, C = D.A, D.B, D.C

cap(x) = isfinite(x) ? clamp(x, 1e-300, 1e300) : 1e300   # for log axes
const SCOLORS = Dict(0 => :dodgerblue, 1 => :seagreen, 2 => :firebrick)

# ---------------------------------------------------------------------------
# Figure 1 — Experiment A: constant-coefficient stability regions |R(z,θ)|
# ---------------------------------------------------------------------------
function fig_A_regions()
    fig = Figure(size = (8.5INCH, 5.6INCH), fontsize = 11)
    Label(fig[0, 1:3], "Experiment A — constant-coefficient stability region  |R(z,θ)|,  z = λΔt";
          fontsize = 13, font = :bold)
    cmap = cgrad([:navy, :deepskyblue, :white, :orange, :darkred], [0, 0.35, 0.5, 0.65, 1])
    for (ri, θ) in enumerate((0.0, 20.0)), (ci, s) in enumerate(A.svals)
        Z = permutedims(A.region[(s, θ)])           # (nx, ny)
        ax = Axis(fig[ri, ci], xlabel = ci == 2 && ri == 2 ? "Re z" : "",
                  ylabel = ci == 1 ? "Im z" : "",
                  title = ri == 1 ? "s=$s  (order $(2s+2))" : "")
        heatmap!(ax, A.xs, A.ys, Z; colormap = cmap, colorrange = (0, 2))
        contour!(ax, A.xs, A.ys, Z; levels = [1.0], color = :black, linewidth = 1.5)
        vlines!(ax, [0.0]; color = (:black, 0.4), linestyle = :dash, linewidth = 0.8)
        ci == 3 && Label(fig[ri, 4], "θ = ωΔt = $(Int(θ))"; rotation = -π/2, tellheight = false)
    end
    Colorbar(fig[1:2, 5]; colormap = cmap, colorrange = (0, 2), label = "|R|")
    Label(fig[3, 1:3],
          "Stable region (|R|≤1, blue) = closed left half-plane; boundary = imaginary axis, " *
          "independent of θ and s.  |R|≡1 on iℝ (norm-preserving).";
          fontsize = 9, color = :gray30)
    save(joinpath(PLOTS, "stab_A_regions.png"), fig; px_per_unit = 2)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 2 — Experiment A: norm preservation on iℝ and (non-)L-stability
# ---------------------------------------------------------------------------
function fig_A_summary()
    fig = Figure(size = (8INCH, 3.4INCH), fontsize = 11)
    ax1 = Axis(fig[1, 1]; xlabel = "θ = ωΔt", ylabel = "sup_ν ‖R(iν,θ)|−1|",
               yscale = log10, title = "Norm preservation on the imaginary axis")
    for s in A.svals
        ys = [cap(A.imax[(s, θ)]) for θ in A.θ_axis]
        scatterlines!(ax1, collect(A.θ_axis), ys; color = SCOLORS[s], label = "s=$s")
    end
    hlines!(ax1, [eps()]; color = :gray, linestyle = :dot)
    axislegend(ax1; position = :lt)

    ax2 = Axis(fig[1, 2]; xlabel = "X  (z = −X, real)", ylabel = "|R(−X, θ=20)|",
               xscale = log10, title = "Behaviour as Re z → −∞  (A- but not L-stable)")
    Xs = [10.0, 100.0, 1e3, 1e6]
    for s in A.svals
        scatterlines!(ax2, Xs, A.linf[s]; color = SCOLORS[s], label = "s=$s")
    end
    hlines!(ax2, [1.0]; color = :gray, linestyle = :dash)
    axislegend(ax2; position = :lb)
    save(joinpath(PLOTS, "stab_A_summary.png"), fig; px_per_unit = 2)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 3 — Experiment B: scalar variable-coefficient growth
# ---------------------------------------------------------------------------
function fig_B()
    fig = Figure(size = (8.5INCH, 6INCH), fontsize = 11)
    Label(fig[0, 1:2], "Experiment B — scalar  u′ = iω₀(1+ε cos Ωt)u  (true |u|≡1): growth factor max_n|u_n|";
          fontsize = 13, font = :bold)
    for (idx, ε) in enumerate((0.25, 0.5, 1.0))
        r, c = divrem(idx - 1, 2) .+ (1, 1)
        ax = Axis(fig[r, c]; xlabel = "ω₀Δt", ylabel = "growth factor",
                  xscale = log10, yscale = log10, title = "ε = $ε")
        for s in B.svals
            scatterlines!(ax, B.coarse, cap.(B.Gp[(s, ε)]); color = SCOLORS[s],
                          label = "s=$s plain", markersize = 5)
            scatterlines!(ax, B.coarse, cap.(B.Gc[(s, ε)]); color = SCOLORS[s],
                          linestyle = :dash, label = "s=$s ctrl", markersize = 5)
        end
        hlines!(ax, [1.0]; color = :gray, linestyle = :dot)
        idx == 1 && axislegend(ax; position = :lt, nbanks = 2, labelsize = 7)
    end
    # trajectory panel
    ax = Axis(fig[2, 2]; xlabel = "t", ylabel = "|u_n|",
              title = @sprintf("trajectory (ε=%.1f, ω₀Δt=%.1f)", B.traj_params.ε, B.traj_params.c))
    for s in B.svals
        lines!(ax, B.tgrid, B.traj[(:plain, s)]; color = SCOLORS[s], label = "s=$s plain")
        lines!(ax, B.tgrid, B.traj[(:controlled, s)]; color = SCOLORS[s], linestyle = :dash)
    end
    hlines!(ax, [1.0]; color = :gray, linestyle = :dot)
    axislegend(ax; position = :lt, labelsize = 7)
    save(joinpath(PLOTS, "stab_B_scalar_variable.png"), fig; px_per_unit = 2)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 4 — Experiment C: non-normal 2×2 stability (the CNOT3 mechanism)
# ---------------------------------------------------------------------------
function fig_C()
    fig = Figure(size = (8.5INCH, 6.2INCH), fontsize = 11)
    Label(fig[0, 1:2], "Experiment C — non-normal 2×2 (diagonal drift + carrier coupling), true ‖ψ‖≡1";
          fontsize = 13, font = :bold)

    # (a) per-step amplification (plain Filon): ‖M‖₂ actual & frozen, ρ(M)
    ax = Axis(fig[1, 1]; xlabel = "ω₁Δt", ylabel = "per-step amplification",
              xscale = log10, yscale = log10, title = "(a) plain ‖Mₙ‖₂ and ρ(Mₙ)")
    for s in C.svals
        scatterlines!(ax, C.coarseM, cap.(C.σA[s]); color = SCOLORS[s], markersize = 4,
                      label = "s=$s ‖M‖₂")
        lines!(ax, C.coarseM, cap.(C.ρA[s]); color = SCOLORS[s], linestyle = :dot)
        lines!(ax, C.coarseM, cap.(C.σF[s]); color = SCOLORS[s], linestyle = :dash)
    end
    hlines!(ax, [1.0]; color = :black, linewidth = 0.8)
    axislegend(ax; position = :lt, labelsize = 7)

    # (b) constant-envelope: growth vs coarseness, plain vs controlled
    ax = Axis(fig[1, 2]; xlabel = "ω₁Δt", ylabel = "max_n ‖ψ_n‖",
              xscale = log10, yscale = log10, title = "(b) growth, constant envelope")
    for s in C.svals
        scatterlines!(ax, C.coarseG, cap.(C.GpC[s]); color = SCOLORS[s], markersize = 4,
                      label = "s=$s plain")
        scatterlines!(ax, C.coarseG, cap.(C.GcC[s]); color = SCOLORS[s], linestyle = :dash,
                      markersize = 4, label = "s=$s ctrl")
    end
    hlines!(ax, [1.0]; color = :gray, linestyle = :dot)
    axislegend(ax; position = :lt, nbanks = 2, labelsize = 6)

    # (c) varying envelope: growth vs coarseness, plain vs controlled
    ax = Axis(fig[2, 1]; xlabel = "ω₁Δt", ylabel = "max_n ‖ψ_n‖",
              xscale = log10, yscale = log10, title = "(c) growth, varying envelope")
    for s in C.svals
        scatterlines!(ax, C.coarseG, cap.(C.GpV[s]); color = SCOLORS[s], markersize = 4)
        scatterlines!(ax, C.coarseG, cap.(C.GcV[s]); color = SCOLORS[s], linestyle = :dash,
                      markersize = 4)
    end
    hlines!(ax, [1.0]; color = :gray, linestyle = :dot)

    # (d) coupling sweep at fixed coarse Δt
    ax = Axis(fig[2, 2]; xlabel = "coupling g", ylabel = "max_n ‖ψ_n‖",
              xscale = log10, yscale = log10,
              title = @sprintf("(d) growth vs coupling (ω₁Δt=%.0f)", C.c_fixed))
    for s in C.svals
        scatterlines!(ax, C.gvals, cap.(C.GpG[s]); color = SCOLORS[s], markersize = 4)
        scatterlines!(ax, C.gvals, cap.(C.GcG[s]); color = SCOLORS[s], linestyle = :dash,
                      markersize = 4)
    end
    hlines!(ax, [1.0]; color = :gray, linestyle = :dot)
    Label(fig[3, 1:2], "solid = plain Filon, dashed = controlled Filon.  " *
          "Controlled absorbs the carrier coupling → stays norm-stable where plain blows up.";
          fontsize = 9, color = :gray30)
    save(joinpath(PLOTS, "stab_C_nonnormal.png"), fig; px_per_unit = 2)
    return fig
end

fig_A_regions(); fig_A_summary(); fig_B(); fig_C()
println("Figures written to ", PLOTS)
println("STAB_PLOT_DONE")
