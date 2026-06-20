module FilonResearch

import Polynomials
import Polynomials: Polynomial
import LinearAlgebra: dot, mul!, axpy!, I, Diagonal, cond, eigvals
import LinearMaps: LinearMap
import IterativeSolvers: gmres
import Krylov
import StaticArrays: SVector, SMatrix

# Subpackage providing time-dependent controlled operators A(t) = Σₖ cₖ(t)·Aₖ.
# Lives at src/ControlledOperators (wired in via [sources] in Project.toml).
using ControlledOperators
# Re-export the subpackage's public API so `using FilonResearch` exposes it.
# `derivative` here is the control derivative from ControlledOperators;
# Polynomials.derivative is unused in this package, so no longer imported above.
export AbstractControl, ConstantControl, FourierControl, FunctionControl, ScaledControl
export CarrierControl, carrier_frequency, envelope
export SumControl, components
export Derivative, DerivativeUpTo, derivative, erase_type
export ControlledOperator, Operator, get_controls, evaluate, evaluate!
export materialize, materialize!

# Toggle for the Taylor series branch in filon_moments.
# When true (default), small-ω moments use a Taylor series to avoid
# catastrophic cancellation in the oscillatory recurrence.
# Set to false to use the oscillatory recurrence for all nonzero ω,
# e.g. to reproduce the blow-up seen in the CNOT3 example:
#   FilonResearch.USE_TAYLOR_MOMENTS[] = false
const USE_TAYLOR_MOMENTS = Ref(true)

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
export filon_timestep, Algorithm1, filon_solve, FilonProblem, get_LHS_mat, filon_timestep_integral_error
include("controlled_operators.jl")
export ControlledFunctionOp, ControlledOp
include("spencer_hardcoded_lhs_rhs.jl")
export Ws_explicit_implicit, S_explicit_implicit_filon, S_explicit_implicit_filon_s0, S_analysis_filon_s0, S_explicit_implicit_filon_s1, S_analysis_filon_s1

# Opt-in per-timestep diagnostics shared by the hard-coded and controlled drivers.
include("solve_stats.jl")
export FilonSolveStats

# Hard-coded Filon method (Appendix A, s = 0,1,2) on the ControlledOperator interface.
include("hardcoded_filon.jl")
export filon_solve_hardcoded, filon_timestep_hardcoded, filon_weight_phases
export StaticFilonWeights, DynamicFilonWeights

# Hard-coded Hermite method (the ω = 0 case of the Filon method), same interface.
include("hardcoded_hermite.jl")
export hermite_solve_hardcoded, hermite_timestep_hardcoded, hermite_weight_phases
export StaticHermiteWeights, DynamicHermiteWeights

# Controlled Filon method (Appendix B, s = 0,1,2): per-control carrier-wave ansatz.
include("controlled_filon.jl")
export controlled_filon_solve, controlled_filon_weights
export StaticControlledFilonWeights, DynamicControlledFilonWeights

# Efficient (generator-form) controlled Filon: dense matvecs scale with the
# number of distinct matrices, not the number of carriers (Appendix B).
include("efficient_controlled_filon.jl")
export efficient_controlled_filon_solve, efficient_controlled_filon_weights
export DynamicEfficientControlledFilonWeights

# Efficient controlled Hermite: the ω = 0 case of the efficient controlled Filon
# method (per-control-Hamiltonian quadrature, no frequency consideration).
include("efficient_controlled_hermite.jl")
export efficient_controlled_hermite_solve, efficient_controlled_hermite_weights
export DynamicEfficientControlledHermiteWeights

end # module FilonResearch
