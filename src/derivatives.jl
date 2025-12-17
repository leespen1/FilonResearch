
"""
For the system of ODEs 

    du/dt = A(t) u

Compute the derivatives 
"""
function compute_derivatives()
end


"""
Compute

    d^m (uv)

for m=0, ..., s.
"""
function general_leibniz_rule(u_derivs, v_derivs)
# TODO add uv_derivs, to enable reusing already computed derivatives
    @assert length(u_derivs) == length(v_derivs) "Must provide same number of derivatives for u and v."
    s = length(u_derivs)-1
    prod_derivs = zeros(ComplexF64, 1+s)
    for m in 0:s
        for k in 0:m
            prod_derivs[1+m] += binomial(m, k) * u_derivs[1+m-k] * v_derivs[1+k]
        end
    end
    return prod_derivs
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

