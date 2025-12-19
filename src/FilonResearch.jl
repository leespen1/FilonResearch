module FilonResearch

import Polynomials
import Polynomials: Polynomial, derivative
import LinearAlgebra: dot

include("filon_weights.jl")
export filon_moments, filon_weights, hardcoded_filon_weights
include("hermite_interpolation.jl")
export hardcoded_hermite_cardinal_polynomials, hermite_interpolating_polynomial, derivative_monomial_matrix
include("explicit_filon_integral.jl")
export explicit_filon_integral
include("filon_dahlquist.jl")
export filon_dahlquist, filon_dahlquist_timestep
include("derivatives.jl")
export general_leibniz_rule, exp_iωt_derivs, dahlquist_derivatives
include("scalar_filon.jl")
export filon_timestep

end # module FilonResearch
