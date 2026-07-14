"""
Shared data plumbing for the CNOT3 paper scripts
(`scripts/cnot3/cnot3_convergence_paper.jl` for figures,
`scripts/cnot3/cnot3_tables_paper.jl` for tables): sweep constants, run
loading, and per-(frame, init) derived columns.  Reads jld2 run data and the
cached Vern9 references only — no ODE solver and no plotting backend, so the
tables script stays Makie-free.

Assumes the DrWatson project is active (`@quickactivate "FilonExperiments"`).
"""

using DataFrames
using LinearAlgebra: norm

include(srcdir("error_analysis.jl"))   # load_vern9_reference

const prefix   = get(ENV, "CNOT3_PREFIX", "cnot3Convergence")
const datapath = datadir(prefix)
# Both initial conditions by default; CNOT3_INIT restricts to one.
const INITS = haskey(ENV, "CNOT3_INIT") ? [ENV["CNOT3_INIT"]] : ["basis", "uniform"]

# Problem size (must match the collected sweep).
const NOSC = 10
const NGUARD = 2
const TMAX = 550.0
const NSAVES = 16

const SVALS = (0, 1, 2)

# Drop diverged coarse runs (upper) and the round-off floor (lower).
const ERROR_WINDOW = (1e-13, 1e1)
in_window(e) = (e .>= ERROR_WINDOW[1]) .& (e .<= ERROR_WINDOW[2])

ic_label(ic) = ic == "basis" ? "Gate Basis IC" :
               ic == "uniform" ? "Uniform IC" : "$(ic) IC"

uref_of(frame, init) = load_vern9_reference(; frame, initialCondition = init, Nosc = NOSC,
                                            Nguard = NGUARD, Tmax = TMAX, nsaves = NSAVES)["uref"]

# All collected runs, with `method` as a Symbol.
function load_cnot3_runs()
    println("Reading ", datapath)
    df_all = collect_results(datapath)
    df_all.method = Symbol.(df_all.method)
    return df_all
end

# Per-(frame, init) DataFrame with the derived columns the figures and tables need.
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
