# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is a [DrWatson](https://juliadynamics.github.io/DrWatson.jl/)
research project for Filon quadrature methods in quantum optimal control —
high-order oscillatory integration techniques for solving time-dependent linear
ODEs with control parameters. The repo root *is* the DrWatson project (named
`FilonExperiments`); numerical experiments, data, and plots live at the top
level, while the reusable method implementation lives as a `dev`'d library
package under `lib/`.

**Domain**: Numerical integration of oscillatory integrals of the form ∫f(x)e^(iωx)dx, applied to ODE solving.

### Repository layout (two environments)

- **Umbrella DrWatson project** at the repo root (`Project.toml` name =
  `FilonExperiments`, no `uuid`). Scripts under `scripts/` do
  `@quickactivate "FilonExperiments"` and pull helpers from `src/` via
  `include(srcdir("..."))`. This environment carries the heavy experiment deps
  (CairoMakie, DataFrames, DrWatson, SlurmClusterManager, …) and `dev`'s the
  two `lib/` packages via `[sources]`.
- **`lib/FilonResearch/`** — the core method package (`FilonResearch`), with its
  own `Project.toml`/`Manifest`, lean deps, `src/`, `test/`, and `examples/`.
  It is `dev`'d by the umbrella but is also usable standalone
  (`julia --project=lib/FilonResearch`).
- **`lib/ControlledOperators/`** — a separate package (`ControlledOperators`,
  with package extensions) `dev`'d by both the umbrella and FilonResearch.

A path-`dev`'d package's own `[sources]` is honored for its deps, so the
umbrella does **not** redeclare FilonResearch's transitive git deps
(`BSplines_jll`, etc.). Manifests are intentionally untracked.

## Math Theory
The latex/pdf document in `FilonProjectOverleaf/` contains my mathematical
description of the Filon method, and other writings (e.g. summaries of numerical
examples I performed). This should be consulted when writing reviewing
mathematical aspects of the implementation of the method, or designing new
features.


## Build and Test Commands

```bash
# Instantiate the umbrella (experiment) environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run a DrWatson experiment script (activates the umbrella project itself)
julia --project=. scripts/cnot3/smoke_test.jl

# --- the FilonResearch library package (lib/FilonResearch) ---

# Library test suite (the meaningful unit tests live here)
julia --project=lib/FilonResearch -e 'using Pkg; Pkg.test()'

# Run a library example
julia --project=lib/FilonResearch lib/FilonResearch/examples/dahlquist.jl
```

> CI (`.github/workflows/CI.yml`) currently runs the umbrella's DrWatson stub
> (`test/runtests.jl`), preserving the pre-restructure behavior. The real
> suites are `lib/FilonResearch/test` and `lib/ControlledOperators/test`.

## Architecture

### Core Components

The method package's source files live under `lib/FilonResearch/src/`.

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

Umbrella DrWatson project (repo root):
- `scripts/` - DrWatson experiment scripts (`@quickactivate "FilonExperiments"`)
- `src/` - experiment helper files pulled in via `include(srcdir("..."))`
- `data/`, `plots/`, `notebooks/`, `_research/` - DrWatson output dirs (gitignored)
- `papers/` - write-ups kept under version control

Library packages:
- `lib/FilonResearch/{src,test,examples}/` - the `FilonResearch` method package
- `lib/ControlledOperators/` - the `ControlledOperators` package

## Coding Conventions

- 4-space indentation
- `snake_case` for functions and files
- Module exports centralized in `lib/FilonResearch/src/FilonResearch.jl`
- No enforced formatter; keep diffs small

## Testing

- Library tests in `lib/FilonResearch/test/`, loaded by its `runtests.jl`
- Name test files `test_*.jl` with one feature per file
- Tests must be deterministic (use seeded PRNG where needed)
- Tests verify mathematical correctness and numerical convergence
- Typical tolerance: 1e-15 (relaxed to 1e-10 for difficult intervals)

## Key Implementation Notes

- Uses `mapreduce(*, +, a, b)` instead of `sum(a .* b)` to avoid length mismatches
- Rescaling option available in `filon_timestep()` for numerical stability on large intervals
- Hermite polynomial hard-coding limited to s ≤ 3
- WIP modules: `manufactured_solution.jl`, `hermite_vars.jl`, `controlled_operators.jl`

## Known Issues and Recent Changes

### Small-ω Taylor branch in `filon_moments` (2026-03-16, WIP)

**Problem**: `filon_moments` in `lib/FilonResearch/src/filon_weights.jl` suffers from catastrophic cancellation when ω is small but nonzero. The oscillatory recurrence computes differences like `sin(w) - w·cos(w) ≈ w³/3`, losing digits when divided by powers of w. This causes Filon(ω≠0) to diverge for s≥1 when effective_ω = freq × 0.5 × dt becomes small (e.g. CNOT3 example with small Kerr frequencies, or any problem at high nsteps).

Spencer's objection (2026-06-10): the CNOT3 divergence is probably not this issue. Filon is likely A-stable, but A-stability does not cover variable-coefficient problems, so blowup at coarse timesteps is a plain stability effect — and errors in the blowup region are meaningless anyway. The cancellation in `filon_moments` may still be real, but the CNOT3 example should not be taken as evidence for it.

**Change**: Added an `elseif abs(w) < 2.0` branch in `filon_moments` that uses a Taylor series: `μ_n(w) = Σ_{k=0}^{30} (iw)^k/k! × (b^{n+k+1} - a^{n+k+1})/(n+k+1)`. This avoids cancellation entirely.

**Status**: The Taylor branch is implemented but the `hardcoded_filon_weights` function still uses the old closed-form formulas and may have the same small-ω issue. The `test_manufactured_polynomial_solution.jl` multi-frequency convergence test (Part 5) was failing for s≥1 before this change was made — it is not yet confirmed whether the Taylor branch resolves those failures or if they are a separate issue.

