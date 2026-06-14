# Plot the oscillatory-problem convergence study (data from osc_compute.jl).
# Per problem: a convergence panel (error vs nsteps) and a work-precision panel
# (error vs wall-clock time).  Run under the umbrella env (CairoMakie) via srun.
using Serialization, CairoMakie, Printf

CairoMakie.activate!()
const INCH = 96
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
results = deserialize(joinpath(REPO, "data", "oscillatory", "osc_data.jls"))
PLOTS = joinpath(REPO, "plots", "oscillatory"); mkpath(PLOTS)

cap(x) = isfinite(x) && x > 0 ? clamp(x, 1e-16, 1e30) : (isfinite(x) ? 1e-16 : 1e30)
const SCOL = Dict(0 => :dodgerblue, 1 => :seagreen, 2 => :firebrick)
TITLES = Dict("parametric" => "P1  parametric flux modulation (2 qubits)",
              "chirp"      => "P2  chirped drive (2-level)",
              "fewcycle"   => "P3  few-cycle Gaussian pulse (2-level)")

# colour = ORDER (so matched-order methods share a colour), linestyle = method
style(m, s) = m === :filon  ? (SCOL[s],   :solid,   :circle,    "Filon s=$s (ord $(2s+2))") :
              m === :cfilon ? (SCOL[s],   :dash,    :utriangle, "cFilon s=$s") :
              m === :gl     ? (SCOL[s-1], :dashdot, :diamond,   "GL order $(2s)") :
                              (:black,    :dot,     :xcross,    "RK4")

function panel_conv(ax, P)
    ns = 2.0 .^ P.es
    for (m, s) in P.methods
        col, ls, mk, lab = style(m, s)
        scatterlines!(ax, ns, cap.(P.err[(m, s)]); color = col, linestyle = ls,
                      marker = mk, markersize = 6, linewidth = 1.4, label = lab)
    end
    # order guides p = 2,4,6 anchored to a mid point
    nf = exp10.(range(log10(ns[1]), log10(ns[end]), length = 50))
    for (p, an) in ((2, 0.3), (4, 0.1), (6, 0.03))
        C = an * ns[max(1, end ÷ 2)]^p
        lines!(ax, nf, C ./ nf .^ p; color = (:gray, 0.5), linestyle = :dot, linewidth = 1)
    end
end

function panel_work(ax, P)
    for (m, s) in P.methods
        col, ls, mk, lab = style(m, s)
        scatterlines!(ax, cap.(P.tim[(m, s)]), cap.(P.err[(m, s)]); color = col,
                      linestyle = ls, marker = mk, markersize = 6, linewidth = 1.4)
    end
end

fig = Figure(size = (9INCH, 10.5INCH), fontsize = 11)
Label(fig[0, 1:2], "Filon / controlled-Filon convergence on no-clean-rotating-frame problems";
      fontsize = 14, font = :bold)
for (ri, P) in enumerate(results)
    Label(fig[ri, 1:2, Top()], TITLES[P.label]; fontsize = 12, padding = (0, 0, 2, 8))
    axc = Axis(fig[ri, 1]; xlabel = "number of timesteps", ylabel = "final-time error",
               xscale = log10, yscale = log10, title = ri == 1 ? "convergence" : "")
    panel_conv(axc, P)
    axw = Axis(fig[ri, 2]; xlabel = "wall-clock time (s)", ylabel = "final-time error",
               xscale = log10, yscale = log10, title = ri == 1 ? "work–precision" : "")
    panel_work(axw, P)
    ri == 1 && axislegend(axc; position = :lb, nbanks = 2, labelsize = 7)
end
save(joinpath(PLOTS, "osc_convergence.png"), fig; px_per_unit = 2)
println("wrote ", joinpath(PLOTS, "osc_convergence.png"))

# ---- ω-scaling figure (fast diagonal ω₀ + slow coupling) ----
om = deserialize(joinpath(REPO, "data", "oscillatory", "osc_omega.jls"))
mstyle = Dict(:filon => (:firebrick, :solid, :circle, "Filon s=2"),
              :cfilon => (:orange, :dash, :utriangle, "cFilon s=2"),
              :gl6 => (:seagreen, :dashdot, :diamond, "GL order 6"),
              :rk4 => (:black, :dot, :xcross, "RK4"))
figO = Figure(size = (8.5INCH, 3.8INCH), fontsize = 11)
Label(figO[0, 1:2], "Cost vs carrier ω₀  (fast diagonal + slow coupling); target err $(om.target)";
      fontsize = 12, font = :bold)
axN = Axis(figO[1, 1]; xlabel = "ω₀", ylabel = "min nsteps to target",
           xscale = log10, yscale = log10, title = "step count (all ∝ ω₀)")
for m in om.methods
    c, ls, mk, lab = mstyle[m]
    scatterlines!(axN, om.ω0s, Float64.(om.minN[m]); color = c, linestyle = ls,
                  marker = mk, label = lab)
end
lines!(axN, om.ω0s, om.minN[:filon][1] .* om.ω0s ./ om.ω0s[1];
       color = (:gray, 0.5), linestyle = :dot)            # slope-1 guide
axislegend(axN; position = :lt, labelsize = 8)
axT = Axis(figO[1, 2]; xlabel = "ω₀", ylabel = "wall-clock to target (s)",
           xscale = log10, yscale = log10, title = "wall-clock (Filon cheapest)")
for m in om.methods
    c, ls, mk, _ = mstyle[m]
    scatterlines!(axT, om.ω0s, cap.(om.minT[m]); color = c, linestyle = ls, marker = mk)
end
save(joinpath(PLOTS, "osc_omega.png"), figO; px_per_unit = 2)
println("wrote ", joinpath(PLOTS, "osc_omega.png"))
println("OSC_PLOT_DONE")
