using DrWatson

"""
Compute the stride needed to index hist_fine at the same timepoints as
hist_coarse (assuming the columns of both histories represent values at equally
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
Compute the discrete l2-integral error between two histories, where it is assumed that
hist_fine is a refinement of hist_coarse (i.e. the number of timesteps taken to
produce hist_fine is a multiple of the number of timesteps taken to produce
hist_coarse).
"""
function l2_integral_error_subsample(
    hist_fine::AbstractMatrix{<: Number},
    hist_coarse::AbstractMatrix{<: Number},
    T::Real
)
    
    stride = stride_from_compatible_histories(hist_fine, hist_coarse)
    nsteps_coarse = size(hist_coarse, 2)-1
    dt = T / nsteps_coarse
    err_sq = sum(norm(hist_fine[:, 1 + (k-1)*stride] - hist_coarse[:, k])^2 for k in 1:nsteps_coarse+1)
    return sqrt(dt * err_sq)
end

"""
Given 'fine' and 'coarse' solution histories of an ODE using a method of order
`order` but different numbers of timesteps, approximate the discrete
l2-integral error in the 'fine' solution by using Richardson extrapolation.

Here, hist_fine
"""
function richardson_l2_integral_error(
    hist_fine::AbstractMatrix{<: Number},
    hist_coarse::AbstractMatrix{<: Number},
    nsteps_fine::Integer,
    nsteps_coarse::Integer,
    T::Real,
    order::Integer,
)
    size(hist_fine) == size(hist_coarse) || throw(DimensionMismatch("Sizes of histories are not equal."))
    l2_err = l2_integral_error_subsample(hist_fine, hist_coarse, T)
    r = nsteps_fine / nsteps_coarse
    return l2_err / (r^order - 1)
end

"""
Given 'fine' and 'coarse' solution histories of an ODE using a method of order
`order` but different numbers of timesteps, approximate the discrete
l2-integral error in the 'fine' solution by using Richardson extrapolation.

Infer the number of timesteps from the size of the histories.
"""
function richardson_l2_integral_error_inferred_steps(
    hist_fine::AbstractMatrix{<: Number},
    hist_coarse::AbstractMatrix{<: Number},
    T::Real,
    order::Integer,
)
    l2_err = l2_integral_error_subsample(hist_fine, hist_coarse, T)
    stride = stride_from_compatible_histories(hist_fine, hist_coarse)
    return l2_err / (stride^order - 1)
end

"""
Given 'fine' and 'coarse' solutions of an ODE at a single point in time using a
method of order `order` but different numbers of timesteps, estimate the error
in the 'fine' solution by using Richardson extrapolation.
"""
function richardson_error(
    sol_fine::AbstractVector{<: Number},
    sol_coarse::AbstractVector{<: Number},
    nsteps_fine::Integer,
    nsteps_coarse::Integer,
    order::Integer,
)
    nsteps_fine >= nsteps_coarse || throw(ArgumentError("Must have nsteps_fine ≥ nsteps_coarse."))
    r = nsteps_fine / nsteps_coarse
    return norm(sol_fine .- sol_coarse) / (r^order - 1)
end

"""
Given 'fine' and 'coarse' solutions of an ODE at a single point in time using a
method of order `order` but different numbers of timesteps, estimate the error
in the 'coarse' solution by using Richardson extrapolation.
"""
function richardson_error_coarse(
    sol_fine::AbstractVector{<: Number},
    sol_coarse::AbstractVector{<: Number},
    nsteps_fine::Integer,
    nsteps_coarse::Integer,
    order::Integer,
)
    nsteps_fine >= nsteps_coarse || throw(ArgumentError("Must have nsteps_fine ≥ nsteps_coarse."))
    r = nsteps_fine / nsteps_coarse
    return norm(sol_fine .- sol_coarse) * r^order / (r^order - 1)
end

function downsample_history(history::AbstractMatrix{<: Number}, nsteps::Integer)
    nrows = size(history, 1)
    history_downsampled = similar(history, nrows, 1+nsteps)
    # Getting stride also checks for size incompatibilities
    stride = stride_from_compatible_histories(history, history_downsampled) 
    history_downsampled .= history[:,1:stride:end]
    return history_downsampled
end

"""
Run the convergence simulation defined by config and run_simulation_f. Return
true if the simulation completes sucessfully. Return false otherwise.

It is assumed that config has fields 'nsaves', 'nsteps', and refinementFactor',
and that the output of run_simulation_f has key 'history'.

Rather than just running run_simulation_f directly, this has the advantage of
looking for files corresponding to previous runs of the same simulation with a
different number of timesteps, and using that to do a richardson extrapolation
error estimate and print the results, which is helpful for seeing intermediate
progress, or getting an idea for what the plots will look like before actually
plotting.
"""
function process_convergence_config(run_simulation_f, config, prefix, outdir)
    fmt(x) = @sprintf("%3.2e", x)
    # input validation
    if config.nsteps < config.nsaves
        @warn "Number of timesteps is less than number of saves" config.nsteps config.nsaves maxlog=3
        return false
    end
    if rem(config.nsteps, config.nsaves) != 0
        @warn "Number of saves does not divide the number of timesteps" config.nsteps config.nsaves maxlog=3
        return false
    end

    data, file = produce_or_load(
        run_simulation_f,
        config,
        outdir,
        filename = c -> savename(prefix, c, sort=false),
    )
    # If there is a previous solution which this is a refinement of,
    # then approximate the error using Richardson extrapolation.
    prev_config = (config..., nsteps = div(config.nsteps, config.refinementFactor))
    prev_savename = savename(prefix, prev_config, sort=false)
    prev_file = prev_savename * ".jld2"
    prev_path = joinpath(outdir, prev_file)
    if isfile(prev_path)
        prev_data = load(prev_path)
        order = 2*(config.s+1)

        rich_l2_err = richardson_l2_integral_error(
            data["history"], prev_data["history"], 
            config.nsteps, prev_config.nsteps, config.Tmax, order,
        )
    
        rich_final_err = richardson_error(
            data["history"][:,end], prev_data["history"][:,end], 
            config.nsteps, prev_config.nsteps, order,
        )
    else
        rich_l2_err = missing
        rich_final_err = missing
    end

    nsteps_str = "$(config.refinementFactor)^$(round(Int, log(config.refinementFactor, config.nsteps)))"
    @printf(
        "pid=%d method=%-8s s=%-2d nsteps=%-6s final_err=%-12s l2_err=%-12s t_elapsed=%-10.4e\n",
        myid(),
        config.method,
        config.s,
        nsteps_str,
        ismissing(rich_final_err) ? "missing" : fmt(rich_final_err),
        ismissing(rich_l2_err)   ? "missing" : fmt(rich_l2_err),
        data["t_elapsed"],
    )

    return true
end
