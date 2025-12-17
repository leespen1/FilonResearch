"""
Perform the timestep

    u(1) \\approx u(-1) + \\int_{-1}^{1} \\dot{u} dt

on the equation

    \\dot{u} = \\lambda(t) u,

using Filon quadrature under the ansatz that u(t) = f(t)e^{i\\omega t}.

- λ_prev: value of λ and its derivatives at t=-1
- λ_next: value of λ and its derivatives at t=1
- u_n: value of u at t=-1.
- ω: ansatz frequency

Order of the method will be inferred from length of λ_prev and λ_next.
"""
function unit_interval_timestep(λ::Number, u_prev::Number, ω::Real, s::Integer=0)
    prev_weights, next_weights = filon_weights(ω, s)

    u̇_prev_derivs = dahlquist_derivatives(λ, u_prev, s)
    # Use 1 in place of u_next, to get the LHS factor (...)*uₙ₊₁ = (...)*uₙ
    u̇_next_derivs = dahlquist_derivatives(λ, 1, s)

    # Compute derivatives of the exp(-iωt) factor
    exp_prev_derivs = exp_iωt_derivs(-ω, t, s)
    exp_next_derivs = exp_iωt_derivs(-ω, t, s)

    f_prev_derivs = general_leibniz_rule(u̇_prev_derivs, exp_prev_derivs)
    f_next_derivs = general_leibniz_rule(u̇_next_derivs, exp_next_derivs)

    u_next = u_prev + dot(f_prev_derivs, prev_weights) / (1 + dot(f_next_derivs, next_weights))
    return u_next
end

"""
Given the dahlquist equation

    \\dot{u} = \\lambda u,

compute the the vector [u̇, ü, …]ᵀ.

n_derivs is the number of derivatives to take of u̇.
"""
function dahlquist_derivatives(λ::Number, u::Number, n_derivs::Integer)
    derivs = zeros(ComplexF64, 1+n_derivs)
    for i in 1:1+n_derivs
        derivs[i] = λ^i * u
    end
    return derivs
end

"""
Given the dahlquist equation

    \\dot{u} = \\lambda(t) u,

with time-dependent λ(t), compute the the vector [u̇, ü, …]ᵀ.

Number of derivatives to take is inferred from length of λ (which contains
the value of λ and its derivatives).
"""
function dahlquist_derivatives(λ::AbstractVector, u::Number, n_derivs::Integer)
    @assert length(λ) > n_derivs "Must provide at least n+1 derivatives of λ to take n derivatives of u̇"
    u_derivs = zeros(ComplexF64, 2+n_derivs)
    u_derivs[1] = u
    for i in 1:1+n_derivs
        # Wasteful, redoes work for each succesive derivative, and allocates lots of vectors
        u̇_derivs = general_leibniz_rule(λ[1:i], u_derivs[1:i])
        u_derivs[1+i] = u̇_derivs[end]
    end
    return u_derivs[2:end]
end

