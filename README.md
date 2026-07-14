# FilonExperiments

Companion code for the paper

> Spencer Lee and Daniel Appelö,
> *Filon Methods for Simulating Highly Oscillatory Controlled Quantum Systems*,
> submitted to SIAM Journal on Scientific Computing, 2026.

The repository contains the method implementations (Filon, Controlled Filon,
and Hermite timestepping for controlled quantum systems) and every script,
parameter, and pipeline used to produce the numerical results in the paper.

## Layout

The repository is a [DrWatson](https://juliadynamics.github.io/DrWatson.jl/)
project (`FilonExperiments`) with the reusable method code split into two
library packages that the project `dev`s by path:

- `lib/FilonResearch/` - the core method package: Filon quadrature weights,
  Hermite interpolation, and the `*_solve` timesteppers used in the paper,
  with its own test suite (`lib/FilonResearch/test/`).
- `lib/ControlledOperators/` - representation of controlled operators
  `A(t) = Σ_k c_k(t) A_k` and their carrier decompositions.
- `scripts/` - the numerical experiments (see the reproduction guide below).
- `src/` - experiment helpers included by the scripts, including the CNOT3
  problem definition and control-pulse coefficients
  (`src/cnot3_hoho_helpers.jl`), so all parameters needed to reproduce the
  results are in this repository.
- `data/`, `plots/` - DrWatson output directories, not under version control;
  scripts create and populate them.

## Setup

The paper's results were produced with Julia 1.12.5. From the repository
root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

All dependencies are declared in the `Project.toml` files; the two
unregistered dependencies (`QuantumGateDesign.jl`, used for the CNOT3 problem
setup and as a baseline integrator, and `BSplines_jll`) are pinned to exact
commits in the `[sources]` sections.

To run the method package's unit tests (mathematical correctness and
convergence orders):

```bash
julia --project=lib/FilonResearch -e 'using Pkg; Pkg.test()'
```

## Reproducing the paper's figures and tables

Every figure and table in the paper is written by one of the scripts below
(figures to `plots/<experiment>/`, LaTeX tables next to them). File names
match the graphics names in the LaTeX source.

| Paper item | Output file | Script |
| --- | --- | --- |
| Rabi oscillator convergence figure | `rabi_convergence_E=0.01_Nperiods=100_omega=0.9_omega0=1.0` | `scripts/rabi_oscillator/rabi_frames_convergence.jl` |
| CNOT3 controls and solution figure | `cnot3_controls_solution` | `scripts/cnot3/cnot3_controls_solution.jl` |
| CNOT3 convergence and work-precision figure | `cnot3_convergence_workprecision_labrwa_basis` | `scripts/cnot3/cnot3_convergence_paper.jl` |
| CNOT3 relative time-to-solution table | `cnot3_speedup_time_lab_basis.tex` | `scripts/cnot3/cnot3_tables_paper.jl` |
| CNOT3 GMRES iterations figure | `cnot3_gmres_labrwa_basis` | `scripts/cnot3/cnot3_convergence_paper.jl` |

### Rabi oscillator (self-contained, minutes on a laptop)

```bash
julia --project=. scripts/rabi_oscillator/rabi_frames_convergence.jl
```

Runs the full lab-frame/RWA-frame convergence study (three methods at orders
2/4/6 against a Vern9 reference), caching results in `data/rabi_oscillator/`
and writing the figure and step-size tables to `plots/rabi_oscillator/`.

### CNOT3 gate problem (three stages)

The CNOT3 experiment simulates the CNOT3 gate problem of the High-Order
Hermite Optimization paper (a CNOT gate on a superconducting system of three
coupled subsystems, state dimension 160) over T = 550 ns; the optimized
B-spline control coefficients are hard-coded in
`src/cnot3_hoho_helpers.jl`. Data collection
and plotting are deliberately separate, and every run is cached with
DrWatson's `produce_or_load`, so re-running any stage only computes what is
missing.

1. **Reference solutions.** Vern9 at tolerance `1e-15` for each frame
   (`rwa`, `norwa`, `lab`) and initial condition (`basis`, `uniform`):

   ```bash
   julia --project=. scripts/cnot3/cnot3_collect_reference.jl
   ```

   Optional `--frame` / `--init` flags narrow the set. The lab-frame
   references are by far the most expensive. Results are cached in
   `data/cnot3_vern9ref/`.

2. **Convergence sweep.** Per-(method, order, frame, nsteps) solves with
   timing and GMRES diagnostics:

   ```bash
   julia --project=. scripts/cnot3/cnot3_convergence_collect_data.jl
   ```

   Flags (`--method`, `--frame`, `--s`, `--nsteps`, and others documented
   in the script header) select subsets. Results are cached in
   `data/cnot3Convergence/`.

   This script and `cnot3_collect_reference.jl` distribute *runs* across
   workers: launched inside a Slurm allocation, they add one worker per
   Slurm task (via `SlurmClusterManager`) and `pmap` the run configurations
   over them. Individual runs are serial — there is no intra-run
   parallelism — so wide allocations of single-CPU tasks (~8 GB each) are
   the right shape. Outside Slurm, everything runs serially in one process.

   The paper's sweep uses `Tmax = 550`, `nsaves = 16`, GMRES tolerances
   `1e-15`, frames `rwa` and `lab` (`norwa` is supported but not part of
   the final data), both initial conditions, and consecutive power-of-two
   step counts. The collected ranges, as exponents e in `nsteps = 2^e`
   (where `basis` / `uniform` differ, both are given):

   | frame | s | ControlledFilon | Filon | Hermite | HermiteQGD |
   |---|---|---|---|---|---|
   | rwa | 0 | 8–21 / 8–23 | 8–23 | 8–23 / 8–26 | 8–23 / 8–26 |
   | rwa | 1 | 8–18 | 8–18 | 8–18 / 8–20 | 8–18 / 8–20 |
   | rwa | 2 | 8–18 | 8–18 | 8–18 | 8–18 |
   | lab | 0 | 10–25 / 10–26 | 10–28 | 16–27 / 16–28 | 16–27 / 16–28 |
   | lab | 1 | 10–21 | 10–21 | 14–27 / 14–29 | 14–26 / 14–29 |
   | lab | 2 | 10–18 | 10–18 | 14–22 / 14–23 | 14–22 / 14–23 |

   To collect a specific slice, pass explicit step counts. For example, the
   lab-frame `ControlledFilon` `s = 1` runs:

   ```bash
   julia --project=. scripts/cnot3/cnot3_convergence_collect_data.jl \
     --frame lab --method ControlledFilon --s 1 \
     --nsteps $(julia -e 'println(join(2 .^ (10:21), ","))')
   ```

   Because runs are cached by their full configuration, sweeps compose:
   repeating a command (or running a superset) computes only what is
   missing. The deepest step counts dominate the cost; the lab-frame
   Hermite runs at 2^29 steps are multi-hour cluster jobs, while a
   workstation can reproduce the moderate-step portion directly.

3. **Figures and tables** (fast; read cached data only):

   ```bash
   julia --project=. scripts/cnot3/cnot3_convergence_paper.jl   # figures
   julia --project=. scripts/cnot3/cnot3_tables_paper.jl        # LaTeX tables
   julia --project=. scripts/cnot3/cnot3_controls_solution.jl   # controls figure
   ```

   Environment toggles: `CNOT3_INIT` restricts to one initial condition,
   `CNOT3_PREFIX` selects an alternate data directory, and `CNOT3_IC_LABEL`
   re-enables the initial-condition suffix in figure titles.

For a quick end-to-end check of the CNOT3 pipeline on a reduced problem, run
`julia --project=. scripts/cnot3/smoke_test.jl`.

### A-stability identities (symbolic check, under a minute)

```bash
julia --project=. scripts/stability/astability_symbolic.jl
```

Verifies, with SymPy, every algebraic identity used in the paper's
A-stability proof for `s = 1`: the quadrature weights derived from the cubic
Hermite cardinals, their real/imaginary decompositions, the collapse of the
first stability condition to `12 φ^8 ≥ 0`, and the perfect-square
factorization of the second. Each check prints `[OK]` when the corresponding
expression simplifies to exactly zero.

### Precollected data

Because the full CNOT3 sweep is expensive, the collected data (references
and per-run results) will be archived with a DOI so that stage 3 can be run
directly; unpack the archive into `data/`. *(Link to be added at
submission.)*

## License

MIT, see `LICENSE`.
