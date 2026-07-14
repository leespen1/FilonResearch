using DrWatson
using Printf
using LinearAlgebra: norm

"""
Compute the stride needed to index `hist_fine` at the same timepoints as
`hist_coarse` (assuming the columns of both histories represent values at equally
spaced timepoints).

Errors if no such stride exists (incompatible sizes).
"""
function stride_from_compatible_histories(
    hist_fine::AbstractMatrix{<: Number},
    hist_coarse::AbstractMatrix{<: Number}
)
    n_rows_fine = size(hist_fine, 1)
    n_rows_coarse = size(hist_coarse, 1)

    nsteps_fine = size(hist_fine, 2) - 1
    nsteps_coarse = size(hist_coarse, 2) - 1
    stride, remainder = divrem(nsteps_fine, nsteps_coarse)

    if n_rows_fine != n_rows_coarse
        throw(DimensionMismatch("Number of rows in hist_fine and hist_coarse are incomparible." *
            " n_rows_fine = $n_rows_fine, n_rows_coarse = $n_rows_coarse."
        ))
    end

    if nsteps_fine < nsteps_coarse
        throw(DimensionMismatch("hist_fine must take more timesteps than hist_coarse." *
            " nsteps_fine = $nsteps_fine, nsteps_coarse = $nsteps_coarse."
        ))
    end

    if remainder != 0
        throw(DimensionMismatch("Number of columns in hist_fine and hist_coarse are incompatible." *
            " nsteps_fine = $nsteps_fine, nsteps_coarse = $nsteps_coarse."
        ))
    end

    return stride
end

"""
Compute the discrete l2-integral error between two histories, where it is assumed
that `hist_fine` is a refinement of `hist_coarse` (i.e. the number of timesteps
taken to produce `hist_fine` is a multiple of the number of timesteps taken to
produce `hist_coarse`).
"""
function l2_integral_error_subsample(
    hist_fine::AbstractMatrix{<: Number},
    hist_coarse::AbstractMatrix{<: Number},
    T::Real
)
    stride = stride_from_compatible_histories(hist_fine, hist_coarse)
    nsteps_coarse = size(hist_coarse, 2) - 1
    dt = T / nsteps_coarse
    err_sq = sum(norm(hist_fine[:, 1 + (k-1)*stride] - hist_coarse[:, k])^2 for k in 1:nsteps_coarse+1)
    return sqrt(dt * err_sq)
end

"""
    load_vern9_reference(; frame, initialCondition, Nosc, Nguard, Tmax, nsaves,
                         abstol = 1e-15, reltol = 1e-15) -> Dict

Load a precomputed Vern9 reference (the dict with `"href"` / `"uref"`) from
`datadir("cnot3_vern9ref")`.  This only reads the cached jld2 — it does *not*
pull in the ODE solver — so the analysis scripts stay lightweight.  Errors if the
reference is missing (run `cnot3_collect_reference.jl` first).

The config keys here must match those `vern9_reference` (in `cnot3_reference.jl`)
writes with, so the `savename` resolves to the same file.
"""
function load_vern9_reference(; frame, initialCondition, Nosc, Nguard, Tmax, nsaves,
                              abstol = 1e-15, reltol = 1e-15)
    cfg = Dict("frame" => string(frame), "initialCondition" => string(initialCondition),
               "abstol" => abstol, "reltol" => reltol, "Nosc" => Nosc,
               "Nguard" => Nguard, "Tmax" => Tmax, "nsaves" => nsaves)
    path = joinpath(datadir("cnot3_vern9ref"), savename("cnot3_vern9ref", cfg, "jld2"))
    isfile(path) || error("Vern9 reference not found:\n  $path\n" *
        "Run scripts/cnot3/cnot3_collect_reference.jl (or its .sb) first.")
    return wload(path)
end

"""
    reference_errors(history, ref, Tmax) -> (final_error, l2_error)

Final-time 2-norm error and discrete l2-integral error of a run `history` against
the Vern9 reference `ref` (the dict from [`load_vern9_reference`](@ref)).  The l2
error needs the full save grid, so it is `missing` for final-only runs (whose
`history` has a single column); the final-time error is always available.
"""
function reference_errors(history, ref, Tmax)
    final_error = norm(history[:, end] .- ref["uref"])
    href = ref["href"]
    l2_error = size(history, 2) == size(href, 2) ?
        l2_integral_error_subsample(history, href, Tmax) : missing
    return (; final_error, l2_error)
end

"""
    process_convergence_config(run_simulation_f, config, prefix, outdir) -> Bool

Run (or load from cache) the convergence simulation defined by `config` and
`run_simulation_f`, writing the result under `outdir` with a `savename` derived
from `config`.  Returns `true` on success.

Caching is via DrWatson's `produce_or_load`, so re-running only computes configs
that are not already on disk.  A one-line progress summary (timing and GMRES
iteration statistics) is printed for each config; errors are *not* computed here
— they are measured downstream against the Vern9 reference solution.

Assumes `config` has fields `nsaves` and `nsteps`, and that the output of
`run_simulation_f` has keys `t_elapsed`, `gmres_mean`, and `gmres_max`.
"""
function process_convergence_config(run_simulation_f, config, prefix, outdir)
    if config.nsteps < config.nsaves
        @warn "Number of timesteps is less than number of saves" config.nsteps config.nsaves maxlog=3
        return false
    end
    if rem(config.nsteps, config.nsaves) != 0
        @warn "Number of saves does not divide the number of timesteps" config.nsteps config.nsaves maxlog=3
        return false
    end

    # tag = false: provenance (gitcommit/gitdirty) is recorded by run_simulation
    # itself, so DrWatson's automatic git tagging would only collide with it.
    data, _ = produce_or_load(
        run_simulation_f,
        config,
        outdir,
        filename = c -> savename(prefix, c, sort=false),
        tag = false,
    )

    fmt(x) = ismissing(x) ? "  --  " : @sprintf("%6.1f", x)
    frame_str = hasproperty(config, :frame) ? " frame=$(rpad(config.frame, 6))" : ""
    @printf(
        "pid=%d method=%-18s%s s=%-2d nsteps=%-9d t_elapsed=%-10.4e gmres(mean/max)=%s/%s\n",
        myid(), config.method, frame_str, config.s, config.nsteps,
        data["t_elapsed"], fmt(data["gmres_mean"]), fmt(data["gmres_max"]),
    )

    return true
end
