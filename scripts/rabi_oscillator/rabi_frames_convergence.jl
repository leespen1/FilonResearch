# =============================================================================
# Rabi oscillator convergence study: lab vs RWA frame × three methods
# =============================================================================
#
# Two-level (Rabi) system  dψ/dt = A(t) ψ = -i H(t) ψ, with
#
#   H_lab(t)   = (ω₀/2) σz + E cos(ω t) σx                           (lab frame)
#
# transformed to the frame rotating at the drive frequency ω, R = e^{-i(ω/2)σz t},
# giving  H̃ = (δ/2) σz + (E/2) σx + (E/2) cos(2ω t) σx - (E/2) sin(2ω t) σy,
# with detuning δ = ω₀ - ω.  Two further frames come from H̃:
#
#   H_rwa(t)   = (δ/2) σz + (E/2) σx                                 (rotating + RWA)
#   H_norwa(t) = H̃                                                   (rotating, no RWA)
#
# The figure shows the lab and RWA panels; the no-RWA frame is the exact rotating-frame
# solution, used here only to compute the RWA modeling error (its accuracy ceiling).
#
# Three methods, each at orders s = 0,1,2 (orders 2/4/6):
#   Filon            ω_ansatz = drift carrier          filon_solve_hardcoded
#   Hermite          ω_ansatz = 0 (carriers folded)    efficient_controlled_hermite_solve
#   Controlled Filon carriers factored (ω+ν)           efficient_controlled_filon_solve
#
# (Regular Hermite — filon_solve_hardcoded with ω_ansatz = 0 — is numerically
#  identical to controlled Hermite, so only the latter is shown, labelled "Hermite".)
#
# Reference: Vern9 at abstol=reltol=1e-13 on each frame's own ODE.
#
# Run:  julia --project=. scripts/rabi_oscillator/rabi_frames_convergence.jl
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using FilonResearch
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEqVerner
using CairoMakie

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

const σx = SMatrix{2,2,ComplexF64}(0, 1, 1, 0)
const σy = SMatrix{2,2,ComplexF64}(0, im, -im, 0)   # [0 -i; i 0]
const σz = SMatrix{2,2,ComplexF64}(1, 0, 0, -1)

# -----------------------------------------------------------------------------
# Frame definitions: each returns (co_fourier, co_carrier, ω_ansatz, A_ref)
#   co_fourier — Hermite / Filon / Controlled Hermite (carrier as Fourier control)
#   co_carrier — Controlled Filon (carriers grouped per matrix as a SumControl)
# -----------------------------------------------------------------------------

function frame_lab(ω₀, ω, E)
    driftM = -im * (ω₀ / 2) * σz
    driveM = -im * E * σx
    cof = ControlledOperator(
        (ConstantControl(1.0), FourierControl(0.0, [1.0], [0.0], ω)),  # 1, cos(ωt)
        (driftM, driveM))
    coc = ControlledOperator(
        (ConstantControl(1.0),
         SumControl(CarrierControl(ConstantControl(0.5), ω),
                    CarrierControl(ConstantControl(0.5), -ω))),        # ½e^{iωt}+½e^{-iωt}
        (driftM, driveM))
    Aref(t) = -im * ((ω₀ / 2) * σz + E * cos(ω * t) * σx)
    return cof, coc, [-ω₀ / 2, ω₀ / 2], Aref
end

function frame_rwa(ω₀, ω, E)
    δ = ω₀ - ω
    driftM = -im * (δ / 2) * σz
    driveM = -im * (E / 2) * σx
    cof = ControlledOperator((ConstantControl(1.0), ConstantControl(1.0)), (driftM, driveM))
    coc = ControlledOperator((ConstantControl(1.0), ConstantControl(1.0)), (driftM, driveM))
    Aref(t) = -im * ((δ / 2) * σz + (E / 2) * σx)
    return cof, coc, [-δ / 2, δ / 2], Aref
end

function frame_norwa(ω₀, ω, E)
    δ = ω₀ - ω
    Az = -im * (δ / 2) * σz
    Ax = -im * σx
    Ay = -im * σy
    cof = ControlledOperator(
        (ConstantControl(1.0),
         FourierControl(E / 2, [E / 2], [0.0], 2ω),      # (E/2)(1 + cos 2ωt)
         FourierControl(0.0, [0.0], [-E / 2], 2ω)),      # -(E/2) sin 2ωt
        (Az, Ax, Ay))
    coc = ControlledOperator(
        (ConstantControl(1.0),
         SumControl(ConstantControl(E / 2 + 0im),
                    CarrierControl(ConstantControl(E / 4 + 0im), 2ω),
                    CarrierControl(ConstantControl(E / 4 + 0im), -2ω)),
         SumControl(CarrierControl(ConstantControl(im * E / 4), 2ω),
                    CarrierControl(ConstantControl(-im * E / 4), -2ω))),
        (Az, Ax, Ay))
    Aref(t) = -im * ((δ / 2) * σz + (E / 2) * σx
                     + (E / 2) * cos(2ω * t) * σx - (E / 2) * sin(2ω * t) * σy)
    return cof, coc, [-δ / 2, δ / 2], Aref
end

# -----------------------------------------------------------------------------
# Solvers and sweep
# -----------------------------------------------------------------------------

const METHODS = (:Filon, :CFilon, :CHermite)       # display / legend order
const METHOD_LABELS = Dict(:Filon => "Filon", :CFilon => "Controlled Filon", :CHermite => "Hermite")
const SVALS = (0, 1, 2)

function solve_method(method, cof, coc, ωans, ψ0, Δt, n, s)
    if method === :Filon
        filon_solve_hardcoded(cof, ψ0, ωans, Δt, n, s; save_final_only = true)
    elseif method === :CHermite
        efficient_controlled_hermite_solve(cof, ψ0, Δt, n, s; save_final_only = true)
    else # :CFilon
        efficient_controlled_filon_solve(coc, ψ0, ωans, Δt, n, s; save_final_only = true)
    end
end

function vern9_ref(Aref, ψ0, T)
    rhs(u, p, t) = Aref(t) * u
    sol = solve(ODEProblem(rhs, SVector{2,ComplexF64}(ψ0), (0.0, T)), Vern9();
                abstol = 1e-13, reltol = 1e-13, save_everystep = false)
    return Vector{ComplexF64}(sol.u[end])
end

"errors[method_symbol][s_index] :: Vector — final-time 2-norm error over nsteps_list."
function run_sweep(framebuilder, ω₀, ω, E, ψ0, T, nsteps_list)
    cof, coc, ωans, Aref = framebuilder(ω₀, ω, E)
    uref = vern9_ref(Aref, ψ0, T)
    errors = Dict(m => [Float64[] for _ in SVALS] for m in METHODS)
    for m in METHODS, (si, s) in enumerate(SVALS), n in nsteps_list
        u = solve_method(m, cof, coc, ωans, ψ0, T / n, n, s)
        push!(errors[m][si], norm(Vector{ComplexF64}(u) .- uref))
    end
    return errors, uref
end

# -----------------------------------------------------------------------------
# Plotting: a single figure with two panels (lab and RWA frames) sharing a y-axis,
# showing the method curves at every order — method by colour, order s by marker — plus
# the RWA accuracy-ceiling line.  The RWA panel omits Controlled Filon (it coincides with
# Filon there).  A shared horizontal legend sits below.
# -----------------------------------------------------------------------------

const METHOD_COLOR = Dict(:Filon    => Makie.wong_colors()[1],   # blue
                          :CFilon   => Makie.wong_colors()[2],   # orange
                          :CHermite => Makie.wong_colors()[3])   # green
const METHOD_LS = Dict(:Filon => :solid, :CFilon => (:dash, 1.0), :CHermite => (:dashdot, 1.0))
const ORDER_MARKERS = (:circle, :rect, :diamond)   # one per s value

"Draw one frame's curves (RWA-error line, method×order scatter) on `ax`, with the timestep
size `Δt` on the x-axis (`dts[i]` pairs with `errors[m][si][i]`).  `methods` selects which
method symbols to draw, e.g. to omit one that coincides with another in this frame."
function plot_frame!(ax, dts, errors, rwa_error; methods = METHODS)
    # RWA accuracy ceiling: the exact rotating-frame and RWA-approximate solutions differ
    # by this fixed amount regardless of timestep, so an RWA computation can never resolve
    # the true physics below this line.
    rwa_error === nothing ||
        hlines!(ax, rwa_error; color = :firebrick, linestyle = :dot, linewidth = 1.5)
    # method = colour + linestyle (so coincident curves stay visible), order s = marker
    for m in methods, si in 1:length(SVALS)
        scatterlines!(ax, dts, errors[m][si];
                      color = METHOD_COLOR[m], linestyle = METHOD_LS[m], marker = ORDER_MARKERS[si],
                      markersize = 7, linewidth = 1.3)
    end
    return ax
end

# Sized for the SIAM single-column paper: \linewidth = \textwidth = 5.125 in.  CairoMakie's
# vector (PDF) unit is 1 pt = 1/72 in, so the figure is built directly in points and saved
# as a PDF at exactly \linewidth, with paper-scale fonts — placed at width=\linewidth it
# needs no rescaling.  The PDF goes to the Overleaf Figures/ dir (fig:rabi-filon-intro);
# a PNG preview goes to the DrWatson plots dir.
const PAPER_PT_PER_IN = 72
const PAPER_WIDTH_IN  = 5.125

"""
Combined convergence figure: lab and RWA panels side by side sharing a y-axis, each
titled, under an overall title, with a shared horizontal legend below.  Saved as a
vector PDF sized for the paper to each path in `pdf_paths`, plus a PNG preview to each
path in `png_paths`.
"""
function make_combined_figure(nsteps_list, errors_lab, errors_rwa, rwa_error, T;
                              ylims = (1e-10, 1e0),
                              pdf_paths = [plotsdir("rabi_oscillator", "rabi_convergence.pdf")],
                              png_paths = [plotsdir("rabi_oscillator", "rabi_convergence.png")])
    ns = collect(Float64, nsteps_list)
    dts = T ./ ns                 # Δt for each run (pairs with the error vectors)
    ypows = -10:2:0
    yticks = (10.0 .^ ypows, [L"10^{%$p}" for p in ypows])

    W = PAPER_WIDTH_IN * PAPER_PT_PER_IN
    # Trim the default 16 pt outer margin to a few pt so the panels fill the page.
    fig = Figure(size = (W, 240), fontsize = 8, figure_padding = (2, 3, 2, 2))
    Label(fig[1, 1:2], "Rabi Oscillator Convergence"; fontsize = 10, font = :bold)

    ax_lab = Axis(fig[2, 1]; title = "Lab Frame", xlabel = L"\Delta t",
                  ylabel = "Final State Error", xscale = log10, yscale = log10,
                  yticks = yticks, limits = (nothing, ylims))
    ax_rwa = Axis(fig[2, 2]; title = "RWA Frame", xlabel = L"\Delta t",
                  xscale = log10, yscale = log10, yticks = yticks, limits = (nothing, ylims))
    plot_frame!(ax_lab, dts, errors_lab, rwa_error)
    # Controlled Filon coincides with regular Filon in the RWA frame (no carriers), so omit it.
    plot_frame!(ax_rwa, dts, errors_rwa, rwa_error; methods = (:Filon, :CHermite))
    linkyaxes!(ax_lab, ax_rwa)
    hideydecorations!(ax_rwa, grid = false)

    method_entries = [LineElement(color = METHOD_COLOR[m], linestyle = METHOD_LS[m], linewidth = 2)
                      for m in METHODS]
    order_entries  = [MarkerElement(marker = ORDER_MARKERS[si], color = :black, markersize = 8)
                      for si in 1:length(SVALS)]
    order_labels = [L"s=%$(s)\;(\mathcal{O}(\Delta t^{%$(2(s+1))}))" for s in SVALS]
    rwa_entry = [LineElement(color = :firebrick, linestyle = :dot, linewidth = 2)]
    # Plain vector literals only — an explicit `Any[...]` trips a Makie legend-text
    # batching bug under theme_latexfonts.  `nbanks = 3` stacks each section into a single
    # column so the three-section legend fits the narrow column width.
    Legend(fig[3, 1:2],
           [method_entries, order_entries, rwa_entry],
           [[METHOD_LABELS[m] for m in METHODS], order_labels, ["RWA Error"]],
           ["Method", "Order", "Modeling"];
           orientation = :horizontal, framevisible = true, titleposition = :left,
           nbanks = 3, patchsize = (14f0, 8f0), colgap = 5, titlegap = 4,
           labelsize = 7, titlesize = 7, padding = (4f0, 4f0, 3f0, 3f0))
    rowgap!(fig.layout, 1, 1)
    rowgap!(fig.layout, 2, 4)

    # pt_per_unit = 1 maps the figure's point-units 1:1 to PDF points, so the page is
    # exactly \linewidth wide and fonts render at their set point size (CairoMakie's
    # default 0.75 would shrink both).
    for p in pdf_paths
        mkpath(dirname(p)); save(p, fig; pt_per_unit = 1); println("  saved → ", p)
    end
    for p in png_paths
        mkpath(dirname(p)); save(p, fig; px_per_unit = 3); println("  saved → ", p)
    end
    return fig
end

# =============================================================================
# Run
# =============================================================================

# Detuned, fast drift and drive (large ω₀, ω ≈ ⅔ω₀, moderate E): opens clean, sustained
# gaps in the lab frame — Filon beats Hermite by factoring the ±ω₀/2 drift carrier, and
# Controlled Filon beats Filon by additionally factoring the full-amplitude ±ω drive
# carrier (most visible at s = 0, roughly 8× and 4× respectively across resolved steps).
ω₀ = 1.0
ω  = 0.9
E  = 0.01
Nperiods = 100
ψ0 = ComplexF64[1.0, 0.0]
T  = Nperiods * 2π / ω₀
nsteps_list = [2^k for k in 4:14]

println("=" ^ 70)
println("Rabi convergence: ω₀=$ω₀, ω=$ω, E=$E, T=$(round(T;digits=3)), nsteps=2^$(4:14)")
println("=" ^ 70)

"Run both frame sweeps and the RWA modeling error, returning one Dict for JLD2 storage."
function collect_rabi_data(ω₀, ω, E, ψ0, T, nsteps_list)
    errors_lab, _ = run_sweep(frame_lab, ω₀, ω, E, ψ0, T, nsteps_list)
    errors_rwa, _ = run_sweep(frame_rwa, ω₀, ω, E, ψ0, T, nsteps_list)
    # RWA modeling error: the exact rotating-frame (no-RWA) and RWA-approximate solutions
    # evolve in the same frame from the same ψ0, differing only by the dropped counter-
    # rotating (±2ω) terms; their separation at T is the RWA's accuracy ceiling — a fixed
    # floor no amount of timestep refinement of the RWA problem can beat.
    Aref_norwa = frame_norwa(ω₀, ω, E)[4]
    Aref_rwa   = frame_rwa(ω₀, ω, E)[4]
    rwa_error  = norm(vern9_ref(Aref_rwa, ψ0, T) .- vern9_ref(Aref_norwa, ψ0, T))
    return @strdict errors_lab errors_rwa rwa_error nsteps_list
end

function print_convergence(label, errors)
    println("\n--- $label ---")
    for m in METHODS, (si, s) in enumerate(SVALS)
        e = errors[m][si]
        ords = [log2(e[i] / e[i+1]) for i in 1:length(e)-1]
        best = maximum(filter(isfinite, ords); init = -Inf)
        println("  ", rpad(METHOD_LABELS[m], 19), " s=$s:  ",
                rpad(string(round(e[1]; sigdigits = 2)), 9), " → ",
                rpad(string(round(e[end]; sigdigits = 2)), 9),
                "  best order ≈ ", round(best; digits = 2))
    end
end

# Collect once into a single JLD2 file named by the system parameters; later runs just read
# it back (no git tagging).  Delete the file (or data/rabi_oscillator/) to force recollection.
# The same parameter savename is reused for the plot files, so data and figures stay paired.
config = Dict("omega0" => ω₀, "omega" => ω, "E" => E, "Nperiods" => Nperiods)
data, datafile = produce_or_load(config, datadir("rabi_oscillator");
                                 prefix = "rabi_convergence", tag = false) do _
    collect_rabi_data(ω₀, ω, E, ψ0, T, nsteps_list)
end
println("data: ", datafile)

errors_lab = data["errors_lab"]
errors_rwa = data["errors_rwa"]
rwa_error  = data["rwa_error"]
println("RWA modeling error (RWA vs exact rotating-frame solution at T): ",
        round(rwa_error; sigdigits = 3))
print_convergence("Lab frame", errors_lab)
print_convergence("Rotating frame (RWA)", errors_rwa)
pdf_name = savename("rabi_convergence", config, "pdf")
png_name = savename("rabi_convergence", config, "png")
make_combined_figure(nsteps_list, errors_lab, errors_rwa, rwa_error, T;
                     pdf_paths = [plotsdir("rabi_oscillator", pdf_name),
                                  projectdir("FilonProjectOverleaf", "Figures", pdf_name)],
                     png_paths = [plotsdir("rabi_oscillator", png_name),
                                  projectdir("FilonProjectOverleaf", "FiguresPNG", png_name)])

println("\nDone.")
