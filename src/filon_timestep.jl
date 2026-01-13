"""
Timestep the linear system of ODEs u̇ = Au using the Filon method, assuming that
A is time-independent.
"""
function filon_timestep_static(
    A::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Number},
    t_n::Real,
    dt::Real,
    s::Integer=0
)
    weights_explicit, weights_implicit = filon_weights(frequencies, s, t_n, t_n+dt)

    N = length(u_n)
    rhs = u_n + Algorithm1(A, u_n, frequencies, t_n, s, weights_explicit)
    LHS = LinearMap(
        u -> u - Algorithm1(A, u, frequencies, t_n+dt, s, weights_implicit),
        N, N
    )
    u_np1 = gmres(LHS, rhs)
    return u_np1
end


# TODO Rename this to something more descriptive!
function Algorithm1(
    A::AbstractMatrix{<: Number},
    u::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t::Real,
    s::Integer,
    weights::AbstractVector{<: AbstractVector{<: Number}},
)
    # Compute A, Ȧ, Ä, …, A⁽ˢ⁾
    # derivatives are zero for time-independent A. For time-dependent case, just need to change A_derivs
    A_derivs = [i == 0 ? A : zero(A) for i in 0:s]
    # Compute u̇, ü, …, u⁽ˢ⁺ʲ⁾ (necessary for computing f and its derivatives)
    u_derivs = linear_ode_derivs(A_derivs, u, s)
    # XXX Note the *negative* frequencies, since we are *inverting* the frequency
    # in u(t) to convert to f(t). Also, it makes a difference whether I use -frequencies or -t!
    freq_factor_derivs = exp_iωt_derivs(-frequencies, t, s)
    # Compute f, ḟ, f̈, …, f⁽ˢ⁾
    f_derivs = multiple_general_leibniz_rule(freq_factor_derivs, u_derivs)

    result = similar(u, ComplexF64)
    result .= 0
    for j = 0:s
        modified_f_derivs = [weights[1+j] .* f_k for f_k in f_derivs]
        result += general_leibniz_rule(A_derivs, modified_f_derivs, j)
    end
    return result
end

    
function filon_solve_static(
    A::AbstractMatrix{<: Number},
    u0::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Number},
    T::Real,
    nsteps::Integer,
    s::Integer=0
)
    dt = T / nsteps
    sol = hcat(u0)

    u_n = u0
    for n in 1:nsteps
        t_n = (n-1)*dt
        u_np1 = filon_timestep_static(A, u_n, frequencies, t_n, dt, s)
        sol = hcat(sol, u_np1)
        u_n = u_np1
    end
    return sol
end
