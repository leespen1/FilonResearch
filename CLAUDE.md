# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FilonResearch is a Julia package implementing Filon quadrature methods for quantum optimal control. It provides high-order oscillatory integration techniques for solving time-dependent linear ODEs with control parameters.

**Domain**: Numerical integration of oscillatory integrals of the form ∫f(x)e^(iωx)dx, applied to ODE solving.

## Build and Test Commands

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific example
julia --project=. examples/dahlquist.jl
```

## Architecture

### Core Components

1. **Filon Weights** (`filon_weights.jl`) - Computes quadrature weights b_{k,j}(ω) using moment integrals. Has hard-coded versions for s ∈ {0,1} on [-1,1].

2. **Hermite Interpolation** (`hermite_interpolation.jl`) - Builds interpolating polynomials matching function values and derivatives at interval endpoints. Hard-coded cardinal polynomials for s ∈ {0,1,2,3}.

3. **Derivatives** (`derivatives.jl`) - Implements Leibniz product rule for computing m-th derivative of products via binomial expansion.

4. **Scalar Filon** (`scalar_filon.jl`) - Main user-facing API. `filon_timestep()` solves one time step of du/dt = λ(t)u with oscillatory ansatz u(t) = f(t)e^(iωt).

5. **Hard-Coded 2×2** (`hard_coded_2by2.jl`) - Specialized order-2 integrator for 2×2 matrix systems.

### Module Dependencies

```
filon_weights ← hermite_interpolation
explicit_filon_integral ← filon_weights
scalar_filon ← derivatives, filon_weights, explicit_filon_integral
```

### Directory Structure

- `src/` - Main Julia module and component files
- `test/` - Test files (one feature per `test_*.jl` file)
- `examples/` - Runnable demonstration scripts
- `daniel/` - Ad hoc research scripts (exploratory, not API)

## Coding Conventions

- 4-space indentation
- `snake_case` for functions and files
- Module exports centralized in `src/FilonResearch.jl`
- No enforced formatter; keep diffs small

## Testing

- Tests in `test/` loaded by `test/runtests.jl`
- Name test files `test_*.jl` with one feature per file
- Tests must be deterministic (use seeded PRNG where needed)
- Tests verify mathematical correctness and numerical convergence
- Typical tolerance: 1e-15 (relaxed to 1e-10 for difficult intervals)

## Key Implementation Notes

- Uses `mapreduce(*, +, a, b)` instead of `sum(a .* b)` to avoid length mismatches
- Rescaling option available in `filon_timestep()` for numerical stability on large intervals
- Hermite polynomial hard-coding limited to s ≤ 3
- WIP modules: `manufactured_solution.jl`, `hermite_vars.jl`, `controlled_operators.jl`
