"""
This is the best way I could think of to store A(t) and its derivatives along
with the other problem parameters. The problem is that T is usually going to be
a tuple of functions, which are singleton types that will make error reports
messy.

But this is the best type-stable way to do it I can think of while keeping it
as general as possible. Issues like this are why I made the Controls interface
in QuantumGateDesign.jl.
"""
struct FilonProblem{T <: Tuple}
    u0::Vector{ComplexF64}
    tf::Float64
    frequencies::Vector{Float64}
    A_deriv_funcs::T
    function FilonProblem(u0::AbstractVector{<: Number}, tf::Real, frequencies::AbstractVector{<: Real}, A_deriv_funcs::Tuple)
        @assert length(u0) == length(frequencies) "Must provide one frequency for each component of u0."
        @assert all(x -> isa(x, Function), A_deriv_funcs) "Each element of A_deriv_funcs must be a function."
        new{typeof(A_deriv_funcs)}(
            convert(Vector{ComplexF64}, u0),
            convert(Float64, tf),
            convert(Vector{Float64}, frequencies),
            A_deriv_funcs
        )
    end
end

"""
Timestep the linear system of ODEs u̇ = Au using the Filon method, assuming that
A is time-independent.
"""
function filon_timestep(
    A::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Number},
    t_n::Real,
    dt::Real,
    s::Integer=0,
    ;
    rescale::Bool=true
)
    # Compute A, Ȧ, Ä, …, A⁽ˢ⁾
    # derivatives are zero for time-independent A. For time-dependent case, just need to change A_derivs
    A_derivs = [i == 0 ? A : zero(A) for i in 0:s]

    return filon_timestep(A_derivs, A_derivs, u_n, frequencies, t_n, dt, s, rescale=rescale)
end

"""
Timestep the linear system of ODEs u̇ = Au using the Filon method, assuming that
A is time-independent.
"""
function filon_timestep(
    A_derivs_tn::Union{AbstractVector, Tuple},
    A_derivs_tnp1::Union{AbstractVector, Tuple},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Number},
    t_n::Real,
    dt::Real,
    s::Integer=0,
    ;
    rescale::Bool=true
    
)
    if rescale
        modified_frequencies = frequencies .* (0.5*dt)
        weights_explicit, weights_implicit = filon_weights(modified_frequencies, s, -1, 1)
        for i in eachindex(weights_explicit, weights_implicit)
            j = i-1
            weights_explicit[i] .*= cis.(frequencies .* (t_n+0.5*dt)) .* (0.5*dt)^(j+1)
            weights_implicit[i] .*= cis.(frequencies .* (t_n+0.5*dt)) .* (0.5*dt)^(j+1)
        end
    else
        weights_explicit, weights_implicit = filon_weights(frequencies, s, t_n, t_n+dt)
    end

    N = length(u_n)
    rhs = u_n + Algorithm1(A_derivs_tn, u_n, frequencies, t_n, s, weights_explicit)
    LHS = LinearMap(
        u -> u - Algorithm1(A_derivs_tnp1, u, frequencies, t_n+dt, s, weights_implicit),
        N, N
    )
    u_np1 = gmres(LHS, rhs)
    return u_np1
end


# TODO Rename this to something more descriptive!
function Algorithm1(
    A_derivs::Union{AbstractVector, Tuple},
    u::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t::Real,
    s::Integer,
    weights::AbstractVector{<: AbstractVector{<: Number}},
)
    @assert all(A -> hasmethod(*, (typeof(A), typeof(u))), A_derivs) "Type of A_derivs does not support `*` operation with type of u. (typeof(A_derivs)=$(typeof(A_derivs)), typeof(u)=$(typeof(u)))" 
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

    
function filon_solve(
    A::AbstractMatrix{<: Number},
    u0::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Number},
    T::Real,
    nsteps::Integer,
    s::Integer=0,
    ;
    rescale::Bool=true
)
    dt = T / nsteps
    sol = hcat(u0)

    u_n = u0
    for n in 1:nsteps
        t_n = (n-1)*dt
        u_np1 = filon_timestep(A, u_n, frequencies, t_n, dt, s, rescale=rescale)
        sol = hcat(sol, u_np1)
        u_n = u_np1
    end
    return sol
end

function filon_solve(
    A_deriv_funcs::Tuple,
    u0::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Number},
    T::Real,
    nsteps::Integer,
    s::Integer=0
    ;
    rescale::Bool=true
)
    @assert length(A_deriv_funcs) >= s "Must provide at least s=$s derivatives in A_deriv_funcs."
    @assert all(x -> isa(x, Function), A_deriv_funcs) "Each element of A_deriv_funcs must be a function."
    dt = T / nsteps
    sol = hcat(u0) # TODO: this will be type-unstable if u0 is not Vector{ComplexF64}

    u_n = u0
    for n in 1:nsteps
        t_n = (n-1)*dt
        A_derivs_tn = [f(t_n) for f in A_deriv_funcs]
        A_derivs_tnp1 = [f(t_n+dt) for f in A_deriv_funcs]
        u_np1 = filon_timestep(A_derivs_tn, A_derivs_tnp1, u_n, frequencies, t_n, dt, s, rescale=rescale)
        sol = hcat(sol, u_np1)
        u_n = u_np1
    end
    return sol

end

function filon_solve(
    prob::FilonProblem,
    nsteps::Integer,
    s::Integer=0,
    ;
    rescale::Bool=true
)
    return filon_solve(
        prob.A_deriv_funcs,
        prob.u0,
        prob.frequencies,
        prob.tf,
        nsteps,
        s,
        rescale=rescale
    )
end




