"""
Compute

    d^m (uv)

for m=0, ..., s.
"""
function multiple_general_leibniz_rule(u_derivs, v_derivs)
    @assert length(u_derivs) == length(v_derivs) "Must provide same number of derivatives for u and v."
    s = length(u_derivs)-1
    return [general_leibniz_rule(u_derivs, v_derivs, m) for m in 0:s]
end

"""
Compute

    d^s (uv),

given d^m(u) and d^m(v) for m=0,…,s.
"""
function general_leibniz_rule(
# TODO: add tests for this function.
    u_derivs::AbstractVector,
    v_derivs::AbstractVector,
    s::Integer,
    #s=min(length(u_derivs), length(v_derivs))-1, # compute derivatives up to order s
)
    @assert isconcretetype(eltype(u_derivs)) && isconcretetype(eltype(v_derivs)) "Eltypes of u and v derivs must be concrete."
    @assert s >= 0 "Derivative order s=$s must be greater than or equal to zero."
    @assert length(u_derivs) > s && length(v_derivs) > s "Must provide > s derivatives of u and v to compute s derivatives of u*v. (s=$s, length(u_derivs)=$(length(u_derivs)), length(v_derivs)=$(length(v_derivs))"
    if eltype(u_derivs) <: AbstractVector && eltype(v_derivs) <: AbstractVector # For vector products, must broadcast, not standard *
        prod_deriv_m = sum(binomial(s, k) .* u_derivs[1+s-k] .* v_derivs[1+k] for k in 0:s)
    else
        prod_deriv_m = sum(binomial(s, k) * u_derivs[1+s-k] * v_derivs[1+k] for k in 0:s)
    end
    return prod_deriv_m
end


"""
Compute the vector where the i-th entry is the (i-1)-th derivative of exp(iωt)
"""
function exp_iωt_derivs(ω::Real, t::Real, n_derivs::Integer; pi_units=false)
    deriv_vec = zeros(ComplexF64, 1+n_derivs)     
    deriv_vec[1] = pi_units ? cispi(ω*t) : cis(ω*t)
    for i in 1:n_derivs
        deriv_vec[i+1] = im*ω*deriv_vec[i]
    end
    return deriv_vec
end

"""
Compute the derivatives of [e^{i\\omega_1 t}, …, e^{i\\omega_n t}], collect
them in a vector of vectors.
"""
function exp_iωt_derivs(ωs::AbstractVector{<: Real}, t::Real, n_derivs::Integer; pi_units=false)
    nfreq = length(ωs)
    deriv_vec = [zeros(ComplexF64, nfreq) for _ in 0:n_derivs]
    for j in 1:nfreq 
        scalar_deriv_vec = exp_iωt_derivs(ωs[j], t, n_derivs; pi_units=pi_units)
        for i in eachindex(deriv_vec, scalar_deriv_vec)
            deriv_vec[i][j] = scalar_deriv_vec[i]
        end
    end
    return deriv_vec
end

"""
Given A, Ȧ, Ä, …, compute u̇, ü, … for the differential equation

    \\frac{du}{dt} = A u,

using the general leibniz rule (product rule)
"""
function linear_ode_derivs(A_derivs, u::AbstractVector{<: Number}, maxorder=length(A_derivs))
# TODO make a hard-coded version of this for first few orders, e.g. do u̇ = (Ȧ + A²)u
    @assert length(A_derivs) >= maxorder-1 "Must provide ≥ (maxorder-1) derivatives of A to compute s derivatives of dudt = A*u."
    u_derivs = Vector{ComplexF64}[]
    push!(u_derivs, convert(Vector{ComplexF64}, u))
    for m in 0:maxorder-1
        dm_u = general_leibniz_rule(A_derivs, u_derivs, m)
        push!(u_derivs, dm_u)
    end
    return u_derivs
end

function linear_ode_derivs_hardcoded(A_derivs, u, maxorder=length(A_derivs))
    @assert length(A_derivs) >= maxorder-1 "Must provide > maxorder derivatives of A to compute s derivatives of dudt = A*u."
    @assert maxorder <= 3 "Hardcoded version only supports computing up to 3 derivatives."

    u_derivs = [u]
    if maxorder > 0
        A = A_derivs[1]
        u̇ = A*u
        push!(u_derivs, u̇)
        if maxorder > 1
            Ȧ = A_derivs[2]
            ü = Ȧ*u + A*u̇
            push!(u_derivs, ü)
            if maxorder > 2
                Ä = A_derivs[3]
                u⃛ = Ä*u + 2*Ȧ*u̇ + A*ü
                push!(u_derivs, u⃛)
            end
        end
    end

    return u_derivs
end

