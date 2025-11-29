"""
Compute the moments

    \\mu_n(x) = \\int_a^b x^n e^{iwx} dx

and store them in a vector.
"""
function filon_moments(degree::Integer, w::Real, a::Real=-1, b::Real=1)
    @assert degree >= 0 "degree must be non-negative."
    moments::Vector{ComplexF64} = fill(NaN+im*NaN, 1+degree)

    if (w == 0)
        for n in 0:degree
            moments[1+n] = (b^(n+1) - a^(n+1)) / (n+1)
        end
    else
        exp_a = exp(im*w*a)
        exp_b = exp(im*w*b)

        moments[1] = (-im/w)*(exp_b - exp_a)
        for n in 1:degree
            moments[1+n] = (-im/w)*(exp_b*b^n - exp_a*a^n) + (im*n/w)*moments[1+n-1]
        end
    end
    return moments
end

"""
Compute the weights
    
    b_{k,j}(\\omega) = I_\\omega[\\ell_{k,j}] = \\int_a^b \\ell_{k,j}(x)e^{i\\omega x} dx
"""
function filon_weights(s::Integer, w::Real, a::Real=-1, b::Real=1)
    degree = 2*s+1
    moments = filon_moments(degree, w, a, b)

    left_weights::Vector{ComplexF64} = fill(NaN+im*NaN, 1+s)
    right_weights::Vector{ComplexF64} = copy(left_weights)

    fa_derivs = zeros(Int64, 1+s)
    fb_derivs = zeros(Int64, 1+s)
    
    for i in 0:s
        # Setup f⁽ʲ⁾(a) = δᵢⱼ, f⁽ʲ⁾(b) = 0
        fa_derivs .= 0 
        fb_derivs .= 0
        fa_derivs[1+i] = 1
        ℓ_ai = hermite_interpolating_polynomial(a, b, fa_derivs, fb_derivs)
        left_weights[1+i] = sum(moments .* Polynomials.coeffs(ℓ_ai))
         
        # Setup f⁽ʲ⁾(a) = 0, f⁽ʲ⁾(b) = δᵢⱼ
        fa_derivs .= 0
        fb_derivs .= 0
        fb_derivs[1+i] = 1
        ℓ_bi = hermite_interpolating_polynomial(a, b, fa_derivs, fb_derivs)
        right_weights[1+i] = sum(moments .* Polynomials.coeffs(ℓ_bi))
    end

    return left_weights, right_weights
end


