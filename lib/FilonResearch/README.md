# FilonResearch

Filon-type timestepping methods (Filon, Controlled Filon, and Hermite) for
highly oscillatory controlled quantum systems. This is the core method
package behind the numerical experiments in the repository root; see the
top-level README for the paper reference and the reproduction guide.

## Tests

Run with `julia --project=lib/FilonResearch -e 'using Pkg; Pkg.test()'` from
the repository root (or `Pkg.test()` with this directory as the active
project).

| File | Description |
|------|-------------|
| `runtests.jl` | Test suite runner |
| `test_hermite_cardinal_polynomials.jl` | Kronecker delta properties of the hard-coded Hermite cardinal polynomials, s ∈ {0,1,2,3} |
| `test_filon_weights.jl` | Filon moments against analytical indefinite integrals |
| `test_explicit_filon_integration.jl` | Explicit Filon integration for polynomial/rational integrands vs analytical results |
| `test_manufactured_polynomial_solution.jl` | Exactness and convergence on manufactured polynomial×exponential solutions, including multi-frequency systems |
| `test_hardcoded_filon.jl` | Hard-coded Filon on `ControlledOperator` problems: correctness and convergence orders |
| `test_hardcoded_hermite.jl` | Hard-coded Hermite (ω = 0) on `ControlledOperator` problems |
| `test_controlled_filon.jl` | Carrier-resolved Controlled Filon (Appendix B) |
| `test_efficient_controlled_filon.jl` | Efficient (generator-form) Controlled Filon matches `controlled_filon_solve`; convergence orders |
| `test_efficient_controlled_hermite.jl` | Efficient controlled Hermite (the ω = 0 counterpart) |
| `test_efficient_filon.jl` | Efficient (A_k-factored) Filon matches `filon_solve_hardcoded`; convergence orders |
| `test_solve_stats.jl` | `FilonSolveStats` solve diagnostics |
