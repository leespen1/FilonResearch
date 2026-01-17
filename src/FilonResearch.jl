module FilonResearch

import Polynomials
import Polynomials: Polynomial, derivative
import LinearAlgebra: dot
import LinearMaps: LinearMap
import IterativeSolvers: gmres

include("filon_weights.jl")
export filon_moments, filon_weights, hardcoded_filon_weights
include("hermite_interpolation.jl")
export hardcoded_hermite_cardinal_polynomials, hermite_interpolating_polynomial, derivative_monomial_matrix
include("explicit_filon_integral.jl")
export explicit_filon_integral
include("derivatives.jl")
export general_leibniz_rule, multiple_general_leibniz_rule, exp_iωt_derivs,
    dahlquist_derivatives, linear_ode_derivs, linear_ode_derivs_hardcoded
include("scalar_filon.jl")
export filon_timestep
include("hard_coded_2by2.jl")
export filon_timestep_order2_size2, filon_order2_size2
export filon_timestep_order4_size2, filon_order4_size2
include("filon_timestep.jl")
export filon_timestep, Algorithm1, filon_solve, FilonProblem
include("manufactured_solution.jl")
export poly_osc_solution, poly_osc_ode_mat
include("controlled_operators.jl")
export ControlledFunctionOp, ControlledOp

end # module FilonResearch
