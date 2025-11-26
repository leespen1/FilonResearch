module FilonResearch

using Polynomials: Polynomial, derivative

include("weights.jl")
export filon_moments, hermite_cardinal_polynomials, hermite_interpolating_polynomial, derivative_monomial_matrix

end # module FilonResearch
