"""
Compute

    d^m (uv)

for m=0, ..., s.
"""
function multiple_general_leibniz_rule(u_derivs::Vector{<: Number}, v_derivs::Vector{<: Number})
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
    u_derivs,
    v_derivs,
    s=min(length(u_derivs), length(v_derivs))-1, # compute derivatives up to order s
)
    @assert s >= 0 "Derivative order s=$s must be greater than or equal to zero."
    @assert length(u_derivs) > s && length(v_derivs) > s "Must provide > s derivatives of u and v to compute s derivatives of u*v."
    prod_deriv_m = sum(binomial(s, k) * u_derivs[1+s-k] * v_derivs[1+k] for k in 0:s)
    return prod_deriv_m
end


"""
Compute the vector where the i-th entry is the (i-1)-th derivative of exp(iωt)
"""
function exp_iωt_derivs(ω, t, n_derivs; pi_units=false)
    deriv_vec = zeros(ComplexF64, 1+n_derivs)     
    deriv_vec[1] = pi_units ? cispi(ω*t) : cis(ω*t)
    for i in 1:n_derivs
        deriv_vec[i+1] = im*ω*deriv_vec[i]
    end
    return deriv_vec
end

"""
Given A, Ȧ, Ä, …, compute u̇, ü, … for the differential equation

    \\frac{du}{dt} = A u,

using the general leibniz rule (product rule)
"""
function linear_ode_derivs(A_derivs, u, s=length(A_derivs)-1)
# TODO make a hard-coded version of this for first few orders, e.g. do u̇ = (Ȧ + A²)u
    @assert length(A_derivs) > s "Must provide > s derivatives of A to compute s derivatives of dudt = A*u."
    u_derivs = [u]
    for m in 0:s
        dm_u = general_leibniz_rule(A_derivs, u_derivs, m)
        push!(u_derivs, dm_u)
    end
    return u_derivs
end
