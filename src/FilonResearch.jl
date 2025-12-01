module FilonResearch

import Polynomials
using Polynomials: Polynomial, derivative

include("filon_weights.jl")
export filon_moments, filon_weights, hardcoded_filon_weights
include("hermite_interpolation.jl")
export hermite_cardinal_polynomials, hermite_interpolating_polynomial, derivative_monomial_matrix,
include("explicit_filon_integral.jl")
explicit_filon_integral

end # module FilonResearch
