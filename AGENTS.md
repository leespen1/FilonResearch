# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Julia module `FilonResearch` and its component files (e.g., `filon_weights.jl`, `explicit_filon_integral.jl`).
- `test/` contains `runtests.jl` plus focused files like `test_filon_weights.jl`.
- `examples/` includes runnable scripts for experiments and plots; generated figures live in `examples/Plots/`.
- `Project.toml` and `Manifest.toml` define the Julia environment and exact dependency set.
- `daniel/` contains ad hoc research scripts; treat as exploratory rather than API code.

## Build, Test, and Development Commands
- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs dependencies for this repo.
- `julia --project=. -e 'using Pkg; Pkg.test()'` runs the full test suite.
- `julia --project=. examples/dahlquist.jl` runs a specific example script (swap the filename as needed).

## Coding Style & Naming Conventions
- Use Julia’s conventional 4-space indentation.
- Prefer `snake_case` for functions and file names (matches existing `src/` and `test/` layout).
- Keep module exports centralized in `src/FilonResearch.jl`.
- No formatter or linter is enforced; keep diffs small and readable.

## Testing Guidelines
- Tests live in `test/` and are loaded by `test/runtests.jl`.
- Name test files `test_*.jl` and keep each file focused on one feature (e.g., weights, derivatives).
- Add coverage for new numerical routines and edge cases; tests should be deterministic.

## Commit & Pull Request Guidelines
- Recent commits use short, capitalized, descriptive sentences (e.g., “Fixed incorrect weight used for implicit part”).
- Keep commits focused and explain the numerical or algorithmic change in the message body if needed.
- For PRs, include: a concise summary, how to run relevant tests, and example output or plots when behavior changes.

## Configuration Tips
- When sharing results, prefer checked-in scripts under `examples/` instead of notebooks so runs are reproducible.
- Avoid committing large generated artifacts unless they are part of the documented examples.
