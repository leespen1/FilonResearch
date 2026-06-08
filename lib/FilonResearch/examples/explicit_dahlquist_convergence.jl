using FilonResearch

"""
Analytically evaluate the integral

    \\int_a^b e^{\\lambda t} dt
"""
function true_solution(λ, a=-1, b=1)
    return (exp(λ*b) - exp(λ*a)) / λ
end


"""
Evaluate the integral

    \\int_a^b e^{\\lambda t} dt

using Filon integration. This integral can be put into Filon-form

    \\int_a^b f(t)e^{i \\omega t} dt

by taking f(t) = e^{real(\\lambda) t}, \\omega = imag(\\lambda).
"""
function filon_solution(λ, a=-1, b=1, ω=imag(λ), s=0, n_intervals=1)
    result = zero(ComplexF64)

    reλ = real(λ)
    reλ_powers = reλ .^ (0:s)

    Δt = (b-a) / n_intervals
    for i in 1:n_intervals
        tₙ = a + (i-1)*Δt
        tₙ₊₁ = tₙ + Δt

        fₙ_derivs = reλ_powers .* exp(reλ * tₙ)
        fₙ₊₁_derivs = reλ_powers .* exp(reλ * tₙ₊₁)
        result += explicit_filon_integral(ω, s, tₙ, tₙ₊₁, fₙ_derivs, fₙ₊₁_derivs)
    end

    return result
end

"""
Test convergence of explicit Filon when evaluating the integral

    \\int_a^b e^{\\lambda t} dt
"""
function main(λ, a=-1, b=1, ω=imag(λ), s=0, n_refinements=4)
    n_intervals_vec = 2 .^ (0:n_refinements)

    filon_solutions = filon_solution.(λ, a, b, ω, s, n_intervals_vec)
    errors = abs.(filon_solutions .- true_solution(λ, a, b))
    
    return log2.(n_intervals_vec), log10.(errors)
end


