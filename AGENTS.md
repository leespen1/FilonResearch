# Repository Guidelines

## Project Structure & Module Organization
The repo root is a DrWatson project (`FilonExperiments`); the method code is a
`dev`'d library package under `lib/`.
- `lib/FilonResearch/src/` holds the Julia module `FilonResearch` and its component files (e.g., `filon_weights.jl`, `explicit_filon_integral.jl`).
- `lib/FilonResearch/test/` contains `runtests.jl` plus focused files like `test_filon_weights.jl`.
- `lib/FilonResearch/examples/` includes runnable scripts for experiments and plots.
- `lib/ControlledOperators/` is a separate `dev`'d package.
- `scripts/`, `src/`, `data/`, `plots/` are the umbrella DrWatson project's experiment scripts, helpers (via `srcdir`), and output dirs.
- Each environment has its own `Project.toml`/`Manifest.toml`; Manifests are gitignored.

## Build, Test, and Development Commands
- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` instantiates the umbrella (experiment) environment.
- `julia --project=lib/FilonResearch -e 'using Pkg; Pkg.test()'` runs the library test suite.
- `julia --project=. scripts/cnot3/smoke_test.jl` runs a DrWatson experiment script.
- `julia --project=lib/FilonResearch lib/FilonResearch/examples/dahlquist.jl` runs a library example.

## Coding Style & Naming Conventions
- Use Julia’s conventional 4-space indentation.
- Prefer `snake_case` for functions and file names (matches existing `src/` and `test/` layout).
- Keep module exports centralized in `lib/FilonResearch/src/FilonResearch.jl`.
- No formatter or linter is enforced; keep diffs small and readable.

## Testing Guidelines
- Library tests live in `lib/FilonResearch/test/` and are loaded by its `runtests.jl`.
- Name test files `test_*.jl` and keep each file focused on one feature (e.g., weights, derivatives).
- Add coverage for new numerical routines and edge cases; tests should be deterministic.

## Commit & Pull Request Guidelines
- Recent commits use short, capitalized, descriptive sentences (e.g., “Fixed incorrect weight used for implicit part”).
- Keep commits focused and explain the numerical or algorithmic change in the message body if needed.
- For PRs, include: a concise summary, how to run relevant tests, and example output or plots when behavior changes.

## Configuration Tips
- When sharing results, prefer checked-in scripts (`scripts/` or `lib/FilonResearch/examples/`) instead of notebooks so runs are reproducible.
- Avoid committing large generated artifacts unless they are part of the documented examples.
