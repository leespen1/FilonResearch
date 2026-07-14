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

   Flags (`--method`, `--frame`, and others documented in the script header)
   select subsets. The full sweep in the paper, which extends to 2^30
   timesteps in the lab frame, was collected on a SLURM cluster; the
   `scripts/cnot3/submit_*.sh` and `*.sb` files are the exact campaign
   scripts used. A workstation can reproduce the moderate-step portion of
   the sweep directly. Results are cached in `data/cnot3Convergence/`.

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

### Precollected data

Because the full CNOT3 sweep is expensive, the collected data (references
and per-run results) will be archived with a DOI so that stage 3 can be run
directly; unpack the archive into `data/`. *(Link to be added at
submission.)*

## License

MIT, see `LICENSE`.
