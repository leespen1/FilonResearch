"""
Perform the timestep

    u(t_{n+1}) \\approx u(t_n) + \\int_{t_n}^{t_{n+1}} \\dot{u} dt

on the equation

    \\dot{u} = \\lambda(t) u,

using Filon quadrature under the ansatz that u(t) = f(t)e^{i\\omega t}.

- λ_prev: value of λ and its derivatives at t_n
- λ_next: value of λ and its derivatives at t_{n+1}
- u_prev: value of u at t_n.
- ω: ansatz frequency.
- s: number of derivatives of the integrand to use.

If λs are given as numbers instead of vectors, then it will be assumed λ is
constant in time.

Order of the method will be inferred from length of λ_prev and λ_next.
"""
function filon_timestep(λ::Union{Number, AbstractVector}, ω::Real, u_prev::Number, s::Integer, t_prev::Real, t_next::Real; rescale=false)

    u̇_prev_derivs = dahlquist_derivatives(λ, u_prev, s)
    # Use 1 in place of u_next, to get the LHS factor (...)*uₙ₊₁ = (...)*uₙ
    u̇_next_derivs = dahlquist_derivatives(λ, 1, s)

    # Compute derivatives of the exp(-iωt) factor
    exp_prev_derivs = exp_iωt_derivs(-ω, t_prev, s)
    exp_next_derivs = exp_iωt_derivs(-ω, t_next, s)

    f_prev_derivs = general_leibniz_rule(u̇_prev_derivs, exp_prev_derivs)
    f_next_derivs = general_leibniz_rule(u̇_next_derivs, exp_next_derivs)

    if rescale # Integrate over interval [-1,1] instead of [t_prev, t_next]
        Δt = t_next - t_prev
        t_center = (t_prev + t_next) / 2
        prev_weights, next_weights = filon_weights(0.5*Δt*ω, s, -1, 1)
        # Factor from change of variables which applies to entire interval
        varchange_factor = cis(ω*t_center) * 0.5 * Δt
        # Factor from change of variables which applied to derivatives of f
        varchange_deriv_factors = (0.5*Δt) .^ (0:s)
        
        f_prev_derivs .*= varchange_deriv_factors
        f_next_derivs .*= varchange_deriv_factors
        
        rhs = u_prev + varchange_factor * sum(f_prev_derivs .* prev_weights)
        lhs_factor = 1 - varchange_factor * sum(f_next_derivs .* next_weights)
        u_next = rhs / lhs_factor
    else
        prev_weights, next_weights = filon_weights(ω, s, t_prev, t_next)
        rhs = u_prev + sum(f_prev_derivs .* prev_weights)
        lhs_factor = 1 - sum(f_next_derivs .* next_weights)
        u_next = rhs / lhs_factor
    end
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

#=
function filon_dahlquist(λ, ω, s, nsteps, tf, u₀)
    Δt = tf / nsteps
    uₙ::ComplexF64 = u₀
    uₙ₊₁::ComplexF64 = NaN
    for step_i in 1:nsteps
        tₙ = (step_i - 1)*Δt
        tₙ₊₁ = (step_i)*Δt
        uₙ₊₁ = filon_dahlquist_timestep(λ, ω, s, tₙ, tₙ₊₁, uₙ)
        uₙ = uₙ₊₁
    end
    return uₙ₊₁
end
=#
