"""
My version of claude's hardcoded_lhs_rhs.jl. Claude seems to have done it
correctly, but I am implementing it by hand to be sure.

Hard-coded LHS/RHS matrix formulations for the implicit Filon method on systems.

For du/dt = A(t)u with ansatz u_k(t) = f_k(t) exp(i ω_k t), we derive
    S_+ u_{n+1} = S_- u_n
where S_+ and S_- are explicit N×N matrices.

Two approaches are provided for each order:
- `_backslash` variants: form S_+ explicitly, solve with backslash
- `_gmres` variants: apply S_+ as a matrix-vector product, solve with GMRES

The weights are computed on [-1,1] and rescaled to [t_n, t_{n+1}].
"""

function Ws_explicit_implicit(frequencies::AbstractVector{<: Real}, s::Integer, Δt::Real)
    scaled_freqs = frequencies .* (0.5 * Δt)
    weights_explicit, weights_implicit = filon_weights(scaled_freqs, s, -1, 1)

    Ω = Diagonal(frequencies)

    Ws_explicit = Diagonal{ComplexF64, Vector{ComplexF64}}[]
    Ws_implicit = Diagonal{ComplexF64, Vector{ComplexF64}}[]

    for j in 0:s
        W_E = (0.5 * Δt)^(j+1) .* exp(Diagonal(im  .* scaled_freqs .* weights_explicit[1+j]))
        W_I = (0.5 * Δt)^(j+1) .* exp(Diagonal(-im .* scaled_freqs .* weights_implicit[1+j]))
        push!(Ws_explicit, W_E)
        push!(Ws_implicit, W_I)
    end

    return Ws_explicit, Ws_implicit
end

function S_explicit_implicit_filon_s0(
        A_n::AbstractMatrix{<: Number},
        A_np1::AbstractMatrix{<: Number},
        frequencies::AbstractVector{<: Number},
        Δt::Real,
)
    s = 0
    Ws_explicit, Ws_implicit = Ws_explicit_implicit(frequencies, s, Δt)
    W_0_E = Ws_explicit[1]
    W_0_I = Ws_implicit[1]

    S_explicit = I + A_n*W_0_E
    S_implicit = I - A_np1*W_0_I

    return S_explicit, S_implicit
end

function S_explicit_implicit_filon(
        A_n_and_derivs,
        A_np1_and_derivs,
        frequencies::AbstractVector{<: Real},
        Δt::Real,
        s::Integer,
)
    @assert eltype(A_n_and_derivs) <: AbstractMatrix "A_n's must be matrices, got $(eltype(A_n_and_derivs))"
    @assert eltype(A_np1_and_derivs) <: AbstractMatrix "A_np1's must be matrices, got $(eltype(A_np1_and_derivs))"

    if s == 0
        S_explicit, S_implicit = S_explicit_implicit_filon_s0(
            A_n_and_derivs[1],
            A_np1_and_derivs[1],
            frequencies,
            Δt,
        )
    elseif s == 1
        S_explicit, S_implicit = S_explicit_implicit_filon_s1(
            A_n_and_derivs[1],
            A_n_and_derivs[2],
            A_np1_and_derivs[1],
            A_np1_and_derivs[2],
            frequencies,
            Δt,
        )
    else
        throw("s must be ≤ 1 (given value: $s)")
    end

    return S_explicit, S_implicit
end

function S_explicit_implicit_filon_s1(
        A_n::AbstractMatrix{<: Number},
        dA_n::AbstractMatrix{<: Number},
        A_np1::AbstractMatrix{<: Number},
        dA_np1::AbstractMatrix{<: Number},
        frequencies::AbstractVector{<: Real},
        Δt::Real,
)
    s = 1
    Ws_explicit, Ws_implicit = Ws_explicit_implicit(frequencies, s, Δt)
    W_0_E = Ws_explicit[1]
    W_1_E = Ws_explicit[2]
    W_0_I = Ws_implicit[1]
    W_1_I = Ws_implicit[2]
    Ω = Diagonal(frequencies)

    S_explicit = I + A_n*W_0_E + dA_n*W_1_E + A_n*W_1_E*(A_n - im*Ω)
    S_implicit = I - A_np1*W_0_I - dA_np1*W_1_I - A_np1*W_1_I*(A_np1 - im*Ω)

    return S_explicit, S_implicit
end

function S_analysis_filon_s0(
        A_n::AbstractMatrix{<: Number},
        A_np1::AbstractMatrix{<: Number},
        frequencies::AbstractVector{<: Number},
        Δt::Real,
        show_vals=false,
)
    spectral_radius(M) = maximum(abs, eigvals(M))
    S_explicit, S_implicit = S_explicit_implicit_filon_s0(A_n, A_np1, frequencies, Δt)
    if show_vals
        @show cond(S_implicit) cond(S_explicit) spectral_radius(inv(S_implicit)*S_explicit)
    end

    return cond(S_implicit), cond(S_explicit), spectral_radius(inv(S_implicit)*S_explicit)
end 


function S_analysis_filon_s1(
        A_n::AbstractMatrix{<: Number},
        dA_n::AbstractMatrix{<: Number},
        A_np1::AbstractMatrix{<: Number},
        dA_np1::AbstractMatrix{<: Number},
        frequencies::AbstractVector{<: Number},
        Δt::Real,
        show_vals=false,
)
    spectral_radius(M) = maximum(abs, eigvals(M))
    S_explicit, S_implicit = S_explicit_implicit_filon_s1(A_n, dA_n, A_np1, dA_np1, frequencies, Δt)
    if show_vals
        @show cond(S_implicit) cond(S_explicit) spectral_radius(inv(S_implicit)*S_explicit)
    end

    return cond(S_implicit), cond(S_explicit), spectral_radius(inv(S_implicit)*S_explicit)
end 


