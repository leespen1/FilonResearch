"""
Hard-coded LHS/RHS matrix formulations for the implicit Filon method on systems.

For du/dt = A(t)u with ansatz u_k(t) = f_k(t) exp(i ω_k t), we derive
    S_+ u_{n+1} = S_- u_n
where S_+ and S_- are explicit N×N matrices.

Two approaches are provided for each order:
- `_backslash` variants: form S_+ explicitly, solve with backslash
- `_gmres` variants: apply S_+ as a matrix-vector product, solve with GMRES

The weights are computed on [-1,1] and rescaled to [t_n, t_{n+1}].
"""

# ============================================================================
# Weight-phase diagonal vectors
# ============================================================================

"""
Compute the weight-phase diagonal vectors W^E_j and W^I_j for j = 0, ..., s.

Returns (WE, WI) where WE[1+j] and WI[1+j] are length-N vectors whose k-th
entry is  (Δt/2)^{j+1} * exp(±i ω̂_k) * b_{*,j}(ω̂_k),
with ω̂_k = ω_k Δt / 2.
"""
function _weight_phase_vectors(frequencies::AbstractVector{<: Real}, s::Integer, t_n::Real, t_np1::Real)
    dt = t_np1 - t_n
    N = length(frequencies)
    half_dt = 0.5 * dt

    # Compute raw weights on [-1,1] at the scaled frequencies
    scaled_freqs = frequencies .* half_dt
    weights_left, weights_right = filon_weights(scaled_freqs, s, -1, 1)

    # Build rescaled weight-phase vectors
    # W^E_j[k] = (dt/2)^{j+1} * exp(+i ω̂_k) * b_{1,j}(ω̂_k)
    # W^I_j[k] = (dt/2)^{j+1} * exp(-i ω̂_k) * b_{2,j}(ω̂_k)
    phase_E = cis.(scaled_freqs)   # exp(+i ω̂_k)
    phase_I = cis.(-scaled_freqs)  # exp(-i ω̂_k)

    T = promote_type(float(eltype(frequencies)), typeof(float(t_n)), typeof(float(t_np1)))
    WE = Vector{Vector{Complex{T}}}(undef, 1+s)
    WI = Vector{Vector{Complex{T}}}(undef, 1+s)

    for j in 0:s
        scale = half_dt^(j+1)
        WE[1+j] = scale .* phase_E .* weights_left[1+j]
        WI[1+j] = scale .* phase_I .* weights_right[1+j]
    end

    return WE, WI
end

"""
Multiply a full matrix A by a diagonal (given as a vector w), from the right:
    result[i,k] = A[i,k] * w[k]
This is equivalent to A * Diagonal(w) but avoids allocating a Diagonal.
"""
function _mul_right_diag!(result::AbstractMatrix, A::AbstractMatrix, w::AbstractVector)
    N = size(A, 1)
    for k in 1:N
        for i in 1:N
            result[i,k] = A[i,k] * w[k]
        end
    end
    return result
end

function _mul_right_diag(A::AbstractMatrix, w::AbstractVector)
    result = similar(A, promote_type(eltype(A), eltype(w)))
    _mul_right_diag!(result, A, w)
    return result
end

"""
Apply y = A * diag(w) * x (matrix times diagonal times vector).
"""
function _A_diag_x(A::AbstractMatrix, w::AbstractVector, x::AbstractVector)
    return A * (w .* x)
end

# ============================================================================
# s=0 (2nd order): M = A * W_0
# ============================================================================

"""
    filon_timestep_s0_backslash(A_n, A_np1, u_n, frequencies, t_n, t_np1)

Second-order (s=0) Filon timestep using explicit LHS/RHS matrices and backslash.

S_+ = I - A_{n+1} * W^I_0
S_- = I + A_n * W^E_0
"""
function filon_timestep_s0_backslash(
    A_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)
    N = length(u_n)
    WE, WI = _weight_phase_vectors(frequencies, 0, t_n, t_np1)

    T = eltype(WE[1])
    # S_- = I + A_n * diag(WE[1])
    S_minus = Matrix{T}(I, N, N) + _mul_right_diag(A_n, WE[1])

    # S_+ = I - A_{n+1} * diag(WI[1])
    S_plus = Matrix{T}(I, N, N) - _mul_right_diag(A_np1, WI[1])

    return S_plus \ (S_minus * u_n)
end

"""
    filon_timestep_s0_gmres(A_n, A_np1, u_n, frequencies, t_n, t_np1)

Second-order (s=0) Filon timestep using hard-coded LHS application and GMRES.
"""
function filon_timestep_s0_gmres(
    A_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)
    N = length(u_n)
    WE, WI = _weight_phase_vectors(frequencies, 0, t_n, t_np1)

    # RHS = u_n + A_n * diag(WE[1]) * u_n
    rhs = u_n + _A_diag_x(A_n, WE[1], u_n)

    # LHS action: u -> u - A_{n+1} * diag(WI[1]) * u
    LHS = LinearMap(
        u -> u - _A_diag_x(A_np1, WI[1], u),
        N, N
    )

    double_precision_digits = 13 # use 1e-13 precision for Float64, scale for higher or lower precision
    tol = eps(real(eltype(rhs)))^(double_precision_digits/16)
    u_np1, ~ = Krylov.gmres(LHS, rhs, atol=tol, rtol=tol)
    return u_np1
end


# ============================================================================
# s=1 (4th order): M = A*W_0 + Ȧ*W_1 + A*W_1*(A - iΩ)
# ============================================================================

"""
Compute M * u for the s=1 method:
    M * u = A * diag(W_0) * u
          + Ȧ * diag(W_1) * u
          + A * diag(W_1) * (A - iΩ) * u
"""
function _apply_M_s1(
    A::AbstractMatrix, dA::AbstractMatrix,
    W0::AbstractVector, W1::AbstractVector,
    Omega::AbstractVector,  # frequency vector (not diagonal matrix)
    u::AbstractVector,
)
    # Term 1: A * diag(W_0) * u
    result = _A_diag_x(A, W0, u)

    # Term 2: Ȧ * diag(W_1) * u
    result += _A_diag_x(dA, W1, u)

    # Term 3: A * diag(W_1) * (A - iΩ) * u
    #   First compute v = (A - iΩ) * u = A*u - i*Ω.*u
    v = A * u - (im .* Omega .* u)
    #   Then A * diag(W_1) * v
    result += _A_diag_x(A, W1, v)

    return result
end

"""
Form M as an explicit matrix for s=1:
    M = A * diag(W_0) + Ȧ * diag(W_1) + A * diag(W_1) * (A - iΩ)
"""
function _form_M_s1(
    A::AbstractMatrix, dA::AbstractMatrix,
    W0::AbstractVector, W1::AbstractVector,
    Omega::AbstractVector,
)
    N = size(A, 1)
    # A * diag(W_0)
    M = _mul_right_diag(A, W0)

    # + Ȧ * diag(W_1)
    M += _mul_right_diag(dA, W1)

    # + A * diag(W_1) * (A - iΩ)
    # First form diag(W_1) * (A - iΩ): diagonal times full matrix
    # [diag(W_1) * B]_{pk} = W_1[p] * B[p,k]
    AminusiOmega = copy(A)
    for k in 1:N
        AminusiOmega[k,k] -= im * Omega[k]
    end
    T = promote_type(eltype(A), eltype(dA), eltype(W0), eltype(W1), eltype(Omega))
    tmp = zeros(T, N, N)
    for k in 1:N
        for p in 1:N
            tmp[p,k] = W1[p] * AminusiOmega[p,k]
        end
    end
    M += A * tmp

    return M
end

"""
    filon_timestep_s1_backslash(A_n, dA_n, A_np1, dA_np1, u_n, frequencies, t_n, t_np1)

Fourth-order (s=1) Filon timestep using explicit LHS/RHS matrices and backslash.

S_+ = I - M_I,   S_- = I + M_E
where M = A*W_0 + Ȧ*W_1 + A*W_1*(A - iΩ).
"""
function filon_timestep_s1_backslash(
    A_n::AbstractMatrix{<: Number},
    dA_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    dA_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)
    N = length(u_n)
    WE, WI = _weight_phase_vectors(frequencies, 1, t_n, t_np1)

    M_E = _form_M_s1(A_n, dA_n, WE[1], WE[2], frequencies)
    M_I = _form_M_s1(A_np1, dA_np1, WI[1], WI[2], frequencies)


    T = eltype(WE[1])
    S_minus = Matrix{T}(I, N, N) + M_E
    S_plus  = Matrix{T}(I, N, N) - M_I

    return S_plus \ (S_minus * u_n)
end

"""
    filon_timestep_s1_gmres(A_n, dA_n, A_np1, dA_np1, u_n, frequencies, t_n, t_np1)

Fourth-order (s=1) Filon timestep using hard-coded LHS application and GMRES.
"""
function filon_timestep_s1_gmres(
    A_n::AbstractMatrix{<: Number},
    dA_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    dA_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)
    N = length(u_n)
    WE, WI = _weight_phase_vectors(frequencies, 1, t_n, t_np1)

    # RHS = u_n + M_E * u_n
    rhs = u_n + _apply_M_s1(A_n, dA_n, WE[1], WE[2], frequencies, u_n)

    # LHS action: u -> u - M_I * u
    LHS = LinearMap(
        u -> u - _apply_M_s1(A_np1, dA_np1, WI[1], WI[2], frequencies, u),
        N, N
    )

    double_precision_digits = 13 # use 1e-13 precision for Float64, scale for higher or lower precision
    tol = eps(real(eltype(rhs)))^(double_precision_digits/16)
    u_np1, ~ = Krylov.gmres(LHS, rhs, atol=tol, rtol=tol)
    return u_np1
end


# ============================================================================
# s=2 (6th order):
#   M = A*W_0 + Ȧ*W_1 + A*W_1*(A - iΩ)
#     + Ä*W_2 + 2Ȧ*W_2*(A - iΩ) + A*W_2*D_2
# where D_2 = -Ω² - 2iΩA + Ȧ + A²
# ============================================================================

"""
Compute M * u for the s=2 method.
"""
function _apply_M_s2(
    A::AbstractMatrix, dA::AbstractMatrix, ddA::AbstractMatrix,
    W0::AbstractVector, W1::AbstractVector, W2::AbstractVector,
    Omega::AbstractVector,
    u::AbstractVector,
)
    # s=1 contributions: A*W_0*u + Ȧ*W_1*u + A*W_1*(A-iΩ)*u
    result = _apply_M_s1(A, dA, W0, W1, Omega, u)

    # j=2, ℓ=0: Ä * diag(W_2) * u
    result += _A_diag_x(ddA, W2, u)

    # j=2, ℓ=1: 2 * Ȧ * diag(W_2) * (A - iΩ) * u
    v = A * u - (im .* Omega .* u)  # (A - iΩ) * u
    result += 2.0 .* _A_diag_x(dA, W2, v)

    # j=2, ℓ=2: A * diag(W_2) * D_2 * u
    #   D_2 = -Ω² - 2iΩA + Ȧ + A²
    #   D_2 * u = -Ω².*u - 2i*Ω.*(A*u) + Ȧ*u + A*(A*u)
    Au = A * u
    D2u = -(Omega.^2) .* u - 2im .* Omega .* Au + dA * u + A * Au
    result += _A_diag_x(A, W2, D2u)

    return result
end

"""
Form the D_2 matrix explicitly:
    D_2 = -Ω² - 2iΩA + Ȧ + A²
where Ω multiplies from the left (row scaling).
"""
function _form_D2(A::AbstractMatrix, dA::AbstractMatrix, Omega::AbstractVector)
    N = size(A, 1)
    D2 = dA + A * A  # Ȧ + A²
    for k in 1:N
        for p in 1:N
            D2[p,k] += -2im * Omega[p] * A[p,k]
        end
        D2[k,k] -= Omega[k]^2
    end
    return D2
end

"""
Form M as an explicit matrix for s=2.
"""
function _form_M_s2(
    A::AbstractMatrix, dA::AbstractMatrix, ddA::AbstractMatrix,
    W0::AbstractVector, W1::AbstractVector, W2::AbstractVector,
    Omega::AbstractVector,
)
    N = size(A, 1)

    # s=1 part
    M = _form_M_s1(A, dA, W0, W1, Omega)

    # j=2, ℓ=0: Ä * diag(W_2)
    M += _mul_right_diag(ddA, W2)

    # j=2, ℓ=1: 2 * Ȧ * diag(W_2) * (A - iΩ)
    AminusiOmega = copy(A)
    for k in 1:N
        AminusiOmega[k,k] -= im * Omega[k]
    end
    # diag(W_2) * (A - iΩ): scale rows
    tmp = zeros(promote_type(eltype(W2), eltype(A)), N, N)
    for k in 1:N
        for p in 1:N
            tmp[p,k] = W2[p] * AminusiOmega[p,k]
        end
    end
    M += 2.0 .* (dA * tmp)

    # j=2, ℓ=2: A * diag(W_2) * D_2
    D2 = _form_D2(A, dA, Omega)
    # diag(W_2) * D_2: scale rows
    tmp2 = zeros(promote_type(eltype(W2), eltype(D2)), N, N)
    for k in 1:N
        for p in 1:N
            tmp2[p,k] = W2[p] * D2[p,k]
        end
    end
    M += A * tmp2

    return M
end

"""
    filon_timestep_s2_backslash(A_n, dA_n, ddA_n, A_np1, dA_np1, ddA_np1, u_n, frequencies, t_n, t_np1)

Sixth-order (s=2) Filon timestep using explicit LHS/RHS matrices and backslash.

S_+ = I - M_I,   S_- = I + M_E
where M = A*W_0 + Ȧ*W_1 + A*W_1*(A-iΩ) + Ä*W_2 + 2Ȧ*W_2*(A-iΩ) + A*W_2*D_2.
"""
function filon_timestep_s2_backslash(
    A_n::AbstractMatrix{<: Number},
    dA_n::AbstractMatrix{<: Number},
    ddA_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    dA_np1::AbstractMatrix{<: Number},
    ddA_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)
    N = length(u_n)
    WE, WI = _weight_phase_vectors(frequencies, 2, t_n, t_np1)

    M_E = _form_M_s2(A_n, dA_n, ddA_n, WE[1], WE[2], WE[3], frequencies)
    M_I = _form_M_s2(A_np1, dA_np1, ddA_np1, WI[1], WI[2], WI[3], frequencies)

    T = eltype(WE[1])
    S_minus = Matrix{T}(I, N, N) + M_E
    S_plus  = Matrix{T}(I, N, N) - M_I

    return S_plus \ (S_minus * u_n)
end

"""
    filon_timestep_s2_gmres(A_n, dA_n, ddA_n, A_np1, dA_np1, ddA_np1, u_n, frequencies, t_n, t_np1)

Sixth-order (s=2) Filon timestep using hard-coded LHS application and GMRES.
"""
function filon_timestep_s2_gmres(
    A_n::AbstractMatrix{<: Number},
    dA_n::AbstractMatrix{<: Number},
    ddA_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    dA_np1::AbstractMatrix{<: Number},
    ddA_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)
    N = length(u_n)
    WE, WI = _weight_phase_vectors(frequencies, 2, t_n, t_np1)

    # RHS = u_n + M_E * u_n
    rhs = u_n + _apply_M_s2(A_n, dA_n, ddA_n, WE[1], WE[2], WE[3], frequencies, u_n)

    # LHS action: u -> u - M_I * u
    LHS = LinearMap(
        u -> u - _apply_M_s2(A_np1, dA_np1, ddA_np1, WI[1], WI[2], WI[3], frequencies, u),
        N, N
    )

    double_precision_digits = 13 # use 1e-13 precision for Float64, scale for higher or lower precision
    tol = eps(real(eltype(rhs)))^(double_precision_digits/16)
    u_np1, ~ = Krylov.gmres(LHS, rhs, atol=tol, rtol=tol)
    return u_np1
end


# ============================================================================
# S_plus / S_minus matrix accessors
# ============================================================================

"""
    filon_S_plus_S_minus(A_funcs, frequencies, t_n, t_np1, s)

Return (S_plus, S_minus) as explicit N×N matrices for the Filon timestep
    S_+ u_{n+1} = S_- u_n.

# Arguments
- `A_funcs`: tuple of functions (A, Ȧ, ..., A^(s)) each mapping t -> matrix
- `frequencies`: ansatz frequencies
- `t_n`, `t_np1`: timestep endpoints
- `s`: order parameter (0, 1, or 2)
"""
function filon_S_plus_S_minus(
    A_funcs::Tuple,
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
    s::Integer,
)
    @assert s in (0, 1, 2) "s must be 0, 1, or 2."
    @assert length(A_funcs) >= s + 1 "Must provide at least s+1 derivative functions."

    N = length(frequencies)
    WE, WI = _weight_phase_vectors(frequencies, s, t_n, t_np1)

    if s == 0
        A_n = A_funcs[1](t_n)
        A_np1 = A_funcs[1](t_np1)
        M_E = _mul_right_diag(A_n, WE[1])
        M_I = _mul_right_diag(A_np1, WI[1])
    elseif s == 1
        A_n = A_funcs[1](t_n)
        dA_n = A_funcs[2](t_n)
        A_np1 = A_funcs[1](t_np1)
        dA_np1 = A_funcs[2](t_np1)
        M_E = _form_M_s1(A_n, dA_n, WE[1], WE[2], frequencies)
        M_I = _form_M_s1(A_np1, dA_np1, WI[1], WI[2], frequencies)
    else  # s == 2
        A_n = A_funcs[1](t_n)
        dA_n = A_funcs[2](t_n)
        ddA_n = A_funcs[3](t_n)
        A_np1 = A_funcs[1](t_np1)
        dA_np1 = A_funcs[2](t_np1)
        ddA_np1 = A_funcs[3](t_np1)
        M_E = _form_M_s2(A_n, dA_n, ddA_n, WE[1], WE[2], WE[3], frequencies)
        M_I = _form_M_s2(A_np1, dA_np1, ddA_np1, WI[1], WI[2], WI[3], frequencies)
    end

    T = eltype(WE[1])
    S_minus = Matrix{T}(I, N, N) + M_E
    S_plus  = Matrix{T}(I, N, N) - M_I

    return S_plus, S_minus
end


# ============================================================================
# Convenience solvers (timestepping loops)
# ============================================================================

"""
    filon_solve_hardcoded(A_funcs, u0, frequencies, T, nsteps, s; method=:backslash)

Solve du/dt = A(t)u using the hard-coded LHS/RHS Filon method.

# Arguments
- `A_funcs`: tuple of functions (A, Ȧ, ..., A^(s)) each mapping t -> matrix
- `u0`: initial condition
- `frequencies`: ansatz frequencies
- `T`: final time
- `nsteps`: number of timesteps
- `s`: order parameter (0, 1, or 2 for 2nd, 4th, or 6th order)
- `method`: `:backslash` or `:gmres`
"""
function filon_solve_hardcoded(
    A_funcs::Tuple,
    u0::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    T::Real,
    nsteps::Integer,
    s::Integer;
    method::Symbol=:backslash,
)
    @assert s in (0, 1, 2) "s must be 0, 1, or 2."
    @assert length(A_funcs) >= s + 1 "Must provide at least s+1 derivative functions (A, Ȧ, ..., A^(s))."

    dt = T / nsteps
    u_n = complex.(float.(u0))
    u_saves = [copy(u_n)]

    for n in 1:nsteps
        t_n = dt * (n - 1)
        t_np1 = dt * n

        if s == 0
            A_n = A_funcs[1](t_n)
            A_np1 = A_funcs[1](t_np1)
            if method == :backslash
                u_n = filon_timestep_s0_backslash(A_n, A_np1, u_n, frequencies, t_n, t_np1)
            else
                u_n = filon_timestep_s0_gmres(A_n, A_np1, u_n, frequencies, t_n, t_np1)
            end
        elseif s == 1
            A_n = A_funcs[1](t_n)
            dA_n = A_funcs[2](t_n)
            A_np1 = A_funcs[1](t_np1)
            dA_np1 = A_funcs[2](t_np1)
            if method == :backslash
                u_n = filon_timestep_s1_backslash(A_n, dA_n, A_np1, dA_np1, u_n, frequencies, t_n, t_np1)
            else
                u_n = filon_timestep_s1_gmres(A_n, dA_n, A_np1, dA_np1, u_n, frequencies, t_n, t_np1)
            end
        elseif s == 2
            A_n = A_funcs[1](t_n)
            dA_n = A_funcs[2](t_n)
            ddA_n = A_funcs[3](t_n)
            A_np1 = A_funcs[1](t_np1)
            dA_np1 = A_funcs[2](t_np1)
            ddA_np1 = A_funcs[3](t_np1)
            if method == :backslash
                u_n = filon_timestep_s2_backslash(A_n, dA_n, ddA_n, A_np1, dA_np1, ddA_np1, u_n, frequencies, t_n, t_np1)
            else
                u_n = filon_timestep_s2_gmres(A_n, dA_n, ddA_n, A_np1, dA_np1, ddA_np1, u_n, frequencies, t_n, t_np1)
            end
        end

        push!(u_saves, copy(u_n))
    end

    return u_saves
end
