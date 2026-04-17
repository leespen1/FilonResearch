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
