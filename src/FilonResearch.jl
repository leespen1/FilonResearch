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
include("derivatives.jl")
export general_leibniz_rule, multiple_general_leibniz_rule, exp_iωt_derivs,
    dahlquist_derivatives, linear_ode_derivs
include("scalar_filon.jl")
export filon_timestep
include("hard_coded_2by2.jl")
export filon_timestep_order2_size2, filon_order2_size2

end # module FilonResearch
