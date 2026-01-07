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
    b_E_w1_vec, b_I_w1_vec = filon_weights(frequencies[1], s, t_n, t_np1)
    b_E_w2_vec, b_I_w2_vec = filon_weights(frequencies[2], s, t_n, t_np1)
    @assert size(b_E_w1_vec) == size(b_E_w2_vec) == size(b_I_w1_vec) == size(b_I_w2_vec) == (1,) "Should only have one weight at each point."
    b_E_w1 = b_E_w1_vec[1]
    b_I_w1 = b_I_w1_vec[1]
    b_E_w2 = b_E_w2_vec[1]
    b_I_w2 = b_I_w2_vec[1]

    rhs = zeros(ComplexF64, 2)
    for i in 1:2
        rhs[i] += u_n[i]
        rhs[i] += b_E_w1*A_n[i,1]*cis(-w1*t_n)*u_n[1]
        rhs[i] += b_E_w2*A_n[i,2]*cis(-w2*t_n)*u_n[2] 
    end

    LHS = ComplexF64[1 0;0 1] # Identity matrix
    for i in 1:2
        LHS[i,1] -= b_E_w1*A_np1[i,1]*cis(-w1*t_np1)
        LHS[i,2] -= b_E_w2*A_np1[i,2]*cis(-w2*t_np1)
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
    dt::Real,
)
    A_func = isa(A, Function) ? A : t -> A # Turn matrix A into function A, if needed
    t = 0
    u_n::Vector{ComplexF64} = u0
    u_saves = [u_n]
    while t < T
        t_n = t
        t_np1 = min(t + dt, T)

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
        t = t_np1
    end

    return u_saves
end

    

