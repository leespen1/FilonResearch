"""
Take a hardcoded 2nd-order Filon timestep for a 2×2 system. Intention is to use
as few external functions as possible in order to reduce sources of code incorrectness.

Weights are recomputed for each time interval, instead of rescaling to [-1,1],
which may cause instability.
"""
function filon_timestep_order2_size2(
    A_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)::Vector{ComplexF64}
    @assert length(frequencies) == 2 "For 2×2 system, should only have 2 frequencies."
    @assert length(u_n) == 2 "For 2×2 system, u should be length 2."
    @assert size(A_n) == size(A_np1) == (2,2) "For 2×2 system, A should be 2×2."

    w1, w2 = frequencies

    s = 0
    b_E_w1, b_I_w1 = filon_weights(frequencies[1], s, t_n, t_np1)
    b_E_w2, b_I_w2 = filon_weights(frequencies[2], s, t_n, t_np1)
    @assert size(b_E_w1) == size(b_E_w2) == size(b_I_w1) == size(b_I_w2) == (1,) "Should only have one weight at each point."

    rhs = zeros(ComplexF64, 2)
    A = A_n
    t = t_n
    for i in 1:2
        rhs[i] += u_n[i]
        rhs[i] += b_E_w1[1]*A[i,1]*cis(-w1*t)*u_n[1]
        rhs[i] += b_E_w2[1]*A[i,2]*cis(-w2*t)*u_n[2] 
    end

    # LHS * u_np1 = RHS * u_n

    LHS = ComplexF64[1 0;0 1] # Identity matrix
    A = A_np1
    t = t_np1
    for i in 1:2
        LHS[i,1] -= b_I_w1[1]*A[i,1]*cis(-w1*t)
        LHS[i,2] -= b_I_w2[1]*A[i,2]*cis(-w2*t)
    end
    #@show t_n t_np1 LHS rhs
    u_np1 = LHS \ rhs
    return u_np1
end

function filon_order2_size2(
    A::Union{AbstractMatrix{<: Number}, Function}, # Function A(t) which return value of A at time t
    u0::Vector{<: Number},
    frequencies::Vector{<: Real},
    T::Real,
    nsteps::Integer,
)
    A_func = isa(A, Function) ? A : t -> A # Turn matrix A into function A, if needed
    u_n::Vector{ComplexF64} = u0
    u_saves = [u_n]

    dt = T / nsteps
    for n in 1:nsteps
        t_n = dt*(n-1)
        t_np1 = dt*n

        A_n = A_func(t_n)
        A_np1 = A_func(t_np1)

        u_np1::Vector{ComplexF64} = filon_timestep_order2_size2(
            A_n,
            A_np1,
            u_n,
            frequencies,
            t_n,
            t_np1
        )

        push!(u_saves, u_np1)

        u_n = u_np1
    end

    return u_saves
end

    
"""
Take a hardcoded 2nd-order Filon timestep for a 2×2 system. Intention is to use
as few external functions as possible in order to reduce sources of code incorrectness.

Weights are recomputed for each time interval, instead of rescaling to [-1,1],
which may cause instability.
"""
function filon_timestep_order4_size2(
    A_n::AbstractMatrix{<: Number},
    dA_n::AbstractMatrix{<: Number},
    A_np1::AbstractMatrix{<: Number},
    dA_np1::AbstractMatrix{<: Number},
    u_n::AbstractVector{<: Number},
    frequencies::AbstractVector{<: Real},
    t_n::Real,
    t_np1::Real,
)::Vector{ComplexF64}
    @assert length(frequencies) == 2 "For 2×2 system, should only have 2 frequencies."
    @assert length(u_n) == 2 "For 2×2 system, u should be length 2."
    @assert size(A_n) == size(A_np1) == (2,2) "For 2×2 system, A should be 2×2."

    w1, w2 = frequencies

    du_n = [A_n[1,1]*u_n[1] + A_n[1,2]*u_n[2], A_n[2,1]*u_n[1] + A_n[2,2]*u_n[2]]

    s = 1
    b_E_w1, b_I_w1 = filon_weights(frequencies[1], s, t_n, t_np1)
    b_E_w2, b_I_w2 = filon_weights(frequencies[2], s, t_n, t_np1)
    @assert size(b_E_w1) == size(b_E_w2) == size(b_I_w1) == size(b_I_w2) == (2,) "Should two weights at each point."

    rhs = zeros(ComplexF64, 2)
    A = A_n
    dA = dA_n
    t = t_n
    for i in 1:2
        rhs[i] += u_n[i]
        # s = 0 contribution
        rhs[i] += b_E_w1[1]*A[i,1]*cis(-w1*t)*u_n[1]
        rhs[i] += b_E_w2[1]*A[i,2]*cis(-w2*t)*u_n[2] 
        # s = 1 contribution
        rhs[i] += b_E_w1[2]*dA[i,1]*cis(-w1*t)*u_n[1]
        rhs[i] += b_E_w1[2]*A[i,1]*-w1*im*cis(-w1*t)*u_n[1]
        rhs[i] += b_E_w1[2]*A[i,1]*cis(-w1*t)*du_n[1]

        rhs[i] += b_E_w2[2]*dA[i,2]*cis(-w2*t)*u_n[2]
        rhs[i] += b_E_w2[2]*A[i,2]*-w2*im*cis(-w2*t)*u_n[2]
        rhs[i] += b_E_w2[2]*A[i,2]*cis(-w2*t)*du_n[2]
    end

    # LHS * u_np1 = RHS * u_n

    LHS = ComplexF64[1 0;0 1] # Identity matrix
    A = A_np1
    dA = dA_np1
    t = t_np1
    for i in 1:2
        # s = 0 contribution
        LHS[i,1] -= b_I_w1[1]*A[i,1]*cis(-w1*t)
        LHS[i,2] -= b_I_w2[1]*A[i,2]*cis(-w2*t)

        # s = 1 contribition
        LHS[i,1] -= b_I_w1[2]*dA[i,1]*cis(-w1*t)
        LHS[i,1] -= b_I_w1[2]*A[i,1]*-w1*im*cis(-w1*t)
        # XXX  du[1] part 
        LHS[i,1] -= b_I_w1[2]*A[i,1]*cis(-w1*t)*A[1,1]
        LHS[i,2] -= b_I_w1[2]*A[i,1]*cis(-w1*t)*A[1,2]
    #du_n = [A_n[1,1]*u_n[1] + A_n[1,2]*u_n[2], A_n[2,1]*u_n[1] + A_n[2,2]*u_n[2]]

        LHS[i,2] -= b_I_w2[2]*dA[i,2]*cis(-w2*t)
        LHS[i,2] -= b_I_w2[2]*A[i,2]*-w2*im*cis(-w2*t)
        # XXX  du[2] part 
        LHS[i,1] -= b_I_w2[2]*A[i,2]*cis(-w2*t)*A[2,1]
        LHS[i,2] -= b_I_w2[2]*A[i,2]*cis(-w2*t)*A[2,2]
    end
    #@show t_n t_np1 LHS rhs
    u_np1 = LHS \ rhs
    return u_np1
end

function filon_order4_size2(
    A::Union{AbstractMatrix{<: Number}, Function}, # Function A(t) which return value of A at time t
    dA::Union{AbstractMatrix{<: Number}, Function}, # Function A(t) which return value of A at time t
    u0::Vector{<: Number},
    frequencies::Vector{<: Real},
    T::Real,
    nsteps::Integer,
)
    A_func = isa(A, Function) ? A : t -> A # Turn matrix A into function A, if needed
    dA_func = isa(dA, Function) ? dA : t -> dA
    u_n::Vector{ComplexF64} = u0
    u_saves = [u_n]

    dt = T / nsteps
    for n in 1:nsteps
        t_n = dt*(n-1)
        t_np1 = dt*n

        A_n = A_func(t_n)
        dA_n = dA_func(t_n)
        A_np1 = A_func(t_np1)
        dA_np1 = dA_func(t_np1)

        u_np1::Vector{ComplexF64} = filon_timestep_order4_size2(
            A_n,
            dA_n,
            A_np1,
            dA_np1,
            u_n,
            frequencies,
            t_n,
            t_np1
        )

        push!(u_saves, u_np1)

        u_n = u_np1
    end

    return u_saves
end
