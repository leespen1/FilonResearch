"""
Hard-coded Hermite cardinal polynomials, i.e. the basis functions ℓ_10, ... ,
ℓ_1s such that ℓ^(k)_1j(1) = δ_kj and ℓ^(k)(-1) = 0 for k=0:s.
"""
function hardcoded_hermite_cardinal_polynomials(s::Integer)
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
"""
function hermite_interpolating_polynomial(a::Real, b::Real, fa_derivs::AbstractVector{<: Real}, fb_derivs::AbstractVector{<: Real})
    n_deriv_a = length(fa_derivs)-1
    n_deriv_b = length(fb_derivs)-1
    degree = n_deriv_a + n_deriv_b + 1

    a_mat = derivative_monomial_matrix(a, n_deriv_a, degree)
    b_mat = derivative_monomial_matrix(b, n_deriv_b, degree)


    LHS = vcat(a_mat, b_mat)
    rhs = vcat(fa_derivs, fb_derivs)

    coefficients = LHS \ rhs
    return Polynomial(coefficients)
end
