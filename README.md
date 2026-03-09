# FilonResearch
Experimental repository for hosting code related to the development of Filon methods for quantum optimal control.

## Examples

| File | Description |
|------|-------------|
| `dahlquist.jl` | Simple convergence test for scalar Dahlquist equation (du/dt = λu) with Filon |
| `dahlquist_experiments.jl` | Experiments on Dahlquist with variable ansatz frequency: on-resonance, off-resonance, damped cases |
| `convergence_order_test.jl` | Verifies convergence order 2(s+1) for scalar, matrix, and time-dependent ODEs |
| `manufactured_poly_solution.jl` | Shows Filon gives exact solutions for polynomial×exponential ansatz; checks Hermite convergence |
| `filon_test.jl` | Basic scalar and matrix Filon correctness tests against exact oscillatory solutions |
| `filon_test_timedependent.jl` | Filon on scalar time-dependent problem with exponential-sinusoidal exact solution |
| `scalar_timedep_convergence.jl` | Convergence for scalar ODE with time-dependent coefficients (α + βω cos(ωt)) |
| `2by2_testing.jl` | Filon integrators for 2×2 diagonal systems at order 2 and 4 |
| `multifrequency_system.jl` | 3D system with multiple distinct frequencies and optional time-dependent control |
| `rescale_comparison.jl` | Compares accuracy with/without rescaling to [-1,1]; shows rescaling improves stability |
| `explicit_dahlquist_convergence.jl` | Accuracy of explicit Filon integration for ∫f(x)e^(iωx)dx |
| `rational_polynomial_accuracy.jl` | Explicit Filon accuracy with rational function integrands |
| `hermite_condition_number.jl` | Condition number of Hermite interpolation matrix vs interval endpoints |
| `brute_force_hermite.jl` | Brute-force search for Hermite cardinal polynomial coefficients for s=3 |
| `controlled_operator_validation.jl` | Validates ControlledFunctionOp automatic derivatives match hand-coded ones |
| `bspline_test.jl` | Integrates Filon with QuantumGateDesign for B-spline quantum control problems |
| `rabi_experiments.jl` | Two-level Rabi oscillation: lab frame vs rotating frame (RWA) |
| `cnot3_convergence_comparison.jl` | Filon vs Hermite convergence for CNOT3 gate design (large system, uses QuantumGateDesign) |

## Tests

| File | Description |
|------|-------------|
| `runtests.jl` | Test suite runner |
| `test_hermite_cardinal_polynomials.jl` | Verifies Kronecker delta properties of hard-coded Hermite polynomials for s ∈ {0,1,2,3} |
| `test_filon_weights.jl` | Filon moments against analytical indefinite integrals |
| `test_explicit_filon_integration.jl` | Explicit Filon integration for polynomial/rational integrands vs analytical results |
| `test_scalar_filon.jl` | Scalar Filon timestep on Dahlquist, exact when λ=iω, rescaled and non-rescaled |
| `test_derivatives.jl` | e^(iωt) derivatives and general Leibniz rule |
