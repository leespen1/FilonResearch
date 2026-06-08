"""
Evaluate the integral

    I_\\omega[f] = \\int_a^b f(x)e^{i\\omega x}dx,

using the plain-vanilla Filon method of order 2(1+s), where f(a), f'(a), ...,
f^{(s)}(a) and f(b), f'(b), ... f^{(s)}(b).
"""
function explicit_filon_integral(w::Real, s::Integer, a::Real, b::Real, fa_derivs, fb_derivs)
    @assert length(fa_derivs) == length(fb_derivs) == 1+s "Currently only support same number of derivatives of f(x) at x=a and x=b."
    a_weights, b_weights = filon_weights(w, s, a, b)
    return sum(a_weights .* fa_derivs) + sum(b_weights .* fb_derivs)
end


