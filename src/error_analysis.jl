using DrWatson
using Printf

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
