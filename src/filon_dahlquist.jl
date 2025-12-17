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

"""
Integrate 

    du/dt = \\lambda u

using the Filon method and using the ansatz that u(t) = f(t) e^{i \\omega t}.
"""
function filon_dahlquist_timestep(λ, ω, s, tₙ, tₙ₊₁, uₙ)
    
    tₙ_weights, tₙ₊₁_weights = filon_weights(ω, s)
    # timestep
    #
    Δt = tₙ₊₁ - tₙ
    ω_factors = (-im*ω) .^ (0:s)
    λ_factors = λ .^ (0:s)
    Δt_factors =  (0.5Δt) .^ (0:s)
    exp_derivs = exp(-im*ω*tₙ) .* ω_factors

    # Do the explicit part
    uₙ_derivs = uₙ .* λ_factors
    fₙ_derivatives = general_leibniz_rule(uₙ_derivs, exp_derivs)
    rhs = uₙ + sum(λ .* Δt_factors .* fₙ_derivatives .* tₙ_weights)

    # Do the implicit part
    uₙ₊₁_derivs = λ_factors # Without the factor of u
    fₙ₊₁_derivatives = general_leibniz_rule(uₙ_derivs, exp_derivs) # Without the factor of u
    lhs_factor = 1 + sum(λ .* Δt_factors .* fₙ₊₁_derivatives .* tₙ₊₁_weights)
    uₙ₊₁ = rhs / lhs_factor

    return uₙ₊₁
end

