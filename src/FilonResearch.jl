module FilonResearch

import Polynomials
using Polynomials: Polynomial, derivative

include("filon_weights.jl")
include("hermite_interpolation.jl")
export filon_moments, filon_weights, hermite_cardinal_polynomials, hermite_interpolating_polynomial, derivative_monomial_matrix

end # module FilonResearch
