"""
Compute the moments

    \\mu_n(x) = \\int_a^b x^n e^{iwx} dx

and store them in a vector.

WARNING: unstable at w=0.
"""
function filon_moments(upto_n::Integer, w::Real, a::Real, b::Real)
    @assert upto_n >= 0 "Must at least one moment."
    moments::Vector{ComplexF64} = fill(NaN+im*NaN, 1+upto_n)

    if (w == 0)
        for n in 0:upto_n
            moments[1+n] = (b^(n+1) - a^(n+1)) / (n+1)
        end
        
    else

        exp_a = exp(im*w*a)
        exp_b = exp(im*w*b)

        moments[1] = (-im/w)*(exp_b - exp_a)
        for n in 1:upto_n
            moments[1+n] = (-im/w)*(exp_b*b^n - exp_a*a^n) + (im*n/w)*moments[1+n-1]
        end
    end
    return moments
end

"""
Hard-coded Hermite cardinal polynomials, i.e. the basis functions ℓ_10, ... ,
ℓ_1s such that ℓ^(k)_1j(1) = δ_kj and ℓ^(k)(-1) = 0 for k=0:s.
"""
function hermite_cardinal_polynomials(s::Integer)
    one_plus_x = Polynomial((1,1)) # 1 + x
    one_minus_x = Polynomial((1,-1)) # 1 - x

    if (s < 0) || (s > 3)
        throw(ArgumentError("s must be between 0 and 3; got $s"))
    end

    polynomials = Vector{Polynomial{Float64, :x}}(undef, s+1)
    if (s == 0)
        polynomials[1] = (1/2)*one_plus_x # ℓ_10
    elseif (s == 1)
        polynomials[1] = (1/4) * one_plus_x^2 * Polynomial((2,-1)) # ℓ_10
        polynomials[2] = -(1/4) * one_plus_x^2 * one_minus_x # ℓ_11
    elseif (s == 2)
        polynomials[1] = (1/16) * one_plus_x^3 * Polynomial((8,-9,3)) # ℓ_10
        polynomials[2] = -(1/16) * one_plus_x^3 * one_minus_x * Polynomial((5,-3)) # ℓ_11
        polynomials[3] = (1/16) * one_plus_x^3 * one_minus_x^2 # ℓ_12
    elseif (s == 3)
        # (according to textbook) polynomials[1] = (1/32) * one_plus_x^4 * Polynomial((19,-37,27,-7)) # ℓ_10
        polynomials[1] = (1/32) * one_plus_x^4 * Polynomial((16,-29,20,-5)) # ℓ_10
        polynomials[2] = -(1/32) * one_plus_x^4 * one_minus_x * Polynomial((11,-14,5)) # ℓ_11
        # (according to textbook) polynomials[3] = (1/32) * one_plus_x^4 * one_minus_x^2 * Polynomial((-2,3)) # ℓ_12
        polynomials[3] = (1/32) * one_plus_x^4 * one_minus_x^2 * Polynomial((3,-2)) # ℓ_12
        polynomials[4] = -(1/96) * one_plus_x^4 * one_minus_x^3 # ℓ_13
    else
        throw
    end

    return polynomials
end

"""
    derivative_monomial_matrix(x, n_deriv, degree)

Return the (1+n_deriv)×(1+degree) upper-triangular matrix whose (i,j)-th entry is
the (i-1)-th derivative of x^(j-1), evaluated at `x`.

E.g.

    1 x x^2
    0 1 2x
    0 0 2
"""
function derivative_monomial_matrix(x, n_deriv, degree)
    M = zeros(eltype(x), n_deriv+1, degree+1)

    for i in 1:n_deriv+1
        k = i - 1                     # derivative order
        for j in i:degree+1
            p = j - 1                 # power
            M[i,j] = p < k ? 0 : factorial(p) / factorial(p-k) * x^(p-k)
        end
    end

    return M
end

"""
Create the polynomial which interpolates f(a), f'(a), ..., and f(b), f'(b), ....
"""
function hermite_interpolating_polynomial(a, b, fa_derivs, fb_derivs)
    n_deriv_a = length(fa_derivs)-1
    n_deriv_b = length(fb_derivs)-1
    degree = n_deriv_a + n_deriv_b + 1

    a_mat = derivative_monomial_matrix(a, n_deriv_a, degree)
    b_mat = derivative_monomial_matrix(b, n_deriv_b, degree)


    LHS = vcat(a_mat, b_mat)
    rhs = vcat(fa_derivs..., fb_derivs...)

    coefficients = LHS \ rhs
    return Polynomial(coefficients)
end


"""
Construct the hermite interpolating polynomial using the given points and
derivative values. That is, constructs the polynomial

    p(x) = c0 + c_1x + c_2x^2 + ...

By solving 

    Ax=b

with 

    [1 a a^2 a^3 ... |
    [0 1 2a 3a^2 ... number of vals and derivs at t
    [0 0 2  6a   ... |
A = [...             |
    [1 b b^2 b^3 ... |
    [0 1 2b 3b^2 ... number of vals and derivs at t
    [0 0 2  6b   ... |
    [...             |

x = [c0, c1, c_2, ...]

b = [f(a), f'(a), ..., f(b), f'(b), ...]

WORK IN PROGRESS
"""
function linear_system_hermite_interpolant(a, b, fa_derivs, fb_derivs)
    degree = length(fa_derivs) + length(fa_derivs) - 1
    A = zeros(degree+1, degree+1)
    for i in 1:length(fa_derivs)

          
    end
end

"""

"""
function convert_weights_to_other_side(weights)
    # make tuple (1, -1, 1, -1, ...)
    other_side_weights = map(enumerate(weights)) do (i, w)
        ifelse(isodd(i-1), -1, 1) * conj(w)
    end
    return other_side_weights
end


"""
Tuple version (some care is needed to make sure output is a tuple).
"""
function convert_weights_to_other_side(weights::NTuple{N,T}) where {N,T}
    # make tuple (1, -1, 1, -1, ...)
    sign_tup = ntuple(i -> ifelse(isodd(i-1), -1, 1), N)
    sign_tup = ntuple(i -> (isodd(i-1) ? -1 : 1),  N)
    return sign_tup .* conj.(weights)
end


"""
Generate s=0 (second-order) Filon Weights
"""
function second_order_weights(ω)
    b_20 = zero(ComplexF64) # For type stability

    if ω == 0
        b_20 += 1.0
    else
        #b_10 = im*exp(-im*ω)/ω - im*sin(ω)/ω^2
        b_20 += -im*exp(im*ω)/ω + im*sin(ω)/ω^2
    end
    right_weights = tuple(b_20)
    left_weights = convert_weights_to_other_side(right_weights)
    return left_weights, right_weights
end

"""
Generate s=1 (fourth-order) Filon Weights
"""
function fourth_order_weights(ω)
    b_20 = zero(ComplexF64) # For type stability
    b_21 = zero(ComplexF64)

    if ω == 0
        #b_10 = 1
        b_20 += 1
        #b_11 = 1/3
        b_21 += -1/3 
    else
        #b_10 = exp(-ω*im)im/ω
        #b_10 += 3im*cos(ω)/(ω^3)
        #b_10 += -3im*sin(ω)/(ω^4)
        b_20 += -im*exp(ω*im)/ω
        b_20 += -3im*cos(ω)/(ω^3)
        b_20 += 3im*sin(ω)/(ω^4)

        #b_11 = -exp(-ω*im)/(ω^2)
        #b_11 += im*(2*exp(-im*ω) + exp(im*ω))/(ω^3)
        #b_11 += -3im*sin(ω)/(ω^4)

        b_21 += exp(ω*im)/(ω^2)
        b_21 += im*(exp(-im*ω) + 2*exp(im*ω))/(ω^3)
        b_21 += -3im*sin(ω)/(ω^4)
    end


    right_weights = (b_20, b_21)
    left_weights = convert_weights_to_other_side(right_weights)
    return left_weights, right_weights
end
