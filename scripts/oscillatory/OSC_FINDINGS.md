# Is Filon suitable for problems with no clean rotating frame?

Convergence study of `filon_solve_hardcoded` / `controlled_filon_solve` (s=0,1,2,
orders 2/4/6) on three problems where the oscillation cannot be analytically
transformed away (so an integrator must resolve it directly) — the regime where
Filon's oscillatory ansatz could earn its keep, unlike CNOT3 (where a clean
rotating frame exists and QGD-Hermite wins, see [[cnot3-frame-campaign-results]]).

Scripts: `osc_compute.jl` (→ `data/oscillatory/osc_data.jls`), `osc_plot.jl`
(→ `plots/oscillatory/osc_convergence.png`), `osc_report.jl` (numbers below).
Run on SLURM via `srun`. Self-contained in the umbrella env: subjects are Filon
and controlled-Filon; the competitor is a classical fixed-step **RK4** (order 4);
ground truth is a fine independent RK4 (cross-checked vs fine Filon s=2 to
1e-11–1e-13). Errors are final-time 2-norm (true ‖ψ‖≡1). Control derivatives are
analytic, finite-difference-validated (rel err ≤ 2e-6).

Problems (all small; drift diagonal is constant → Filon ansatz = bare levels):
- **P1 parametric flux modulation** (2 qubits, 4-dim): ω_b(t)=ω_b0+A cos ω_m t,
  static exchange coupling. Rotating away the modulated b†b gives a Bessel comb.
- **P2 chirped drive** (2-level): E₀cos(ω₀t+½βt²) — the carrier sweeps.
- **P3 few-cycle Gaussian** (2-level): E₀e^{-(t-t₀)²/2τ²}cos ω₀t, τ≈1.5 cycles
  (envelope varies on the carrier timescale — no scale separation).

## Results

**Orders confirmed**: Filon and controlled-Filon hit their design orders 2/4/6 on
all three problems, flooring at ~1e-13–1e-14.

**Work–precision (cheapest run reaching the target; Filon s=2 vs RK4):**

| problem    | err≤1e-6  Filon s2 / RK4 | err≤1e-10  Filon s2 / RK4 | speed-up @1e-10 |
|------------|--------------------------|---------------------------|-----------------|
| parametric | 4.0e-4 / 7.0e-4 s        | 1.33e-3 / 5.29e-3 s       | 4.0×            |
| chirp      | 1.8e-4 / 7.1e-4 s        | 1.23e-3 / 5.83e-3 s       | 4.7×            |
| few-cycle  | 1.2e-4 / 4.5e-4 s        | 4.0e-4 / 3.59e-3 s        | 8.9×            |

**Filon s=2 (order 6) is the most cost-effective method on every problem**, beating
RK4 by ~2× (at 1e-6) to ~4–9× (at 1e-10).

**Stability where RK4 fails**: on the chirp, RK4 blows up at coarse steps
(err 2e11 at ω₀Δt≈6), while every Filon/cFilon variant stays bounded — the
oscillation-aware ansatz tolerates ωΔt ≫ the RK stability limit.

**Controlled vs plain Filon**: cFilon improves *per-step* accuracy at coarse steps
(notably the chirp at s=1: 0.063 vs 0.57, ~9×; negligible for parametric), but it
is **less cost-effective than plain Filon** at matched order — its extra carrier
terms cost more than the accuracy buys (e.g. parametric 1e-10: Filon s=2 1.33e-3 s
vs cFilon s=2 2.48e-3 s). For these problems plain Filon at s=2 is the choice.

## Verdict

**Yes — Filon is suitable for these no-clean-rotating-frame problems.** It
converges at full order, is stable where a standard explicit integrator (RK4)
blows up, and is the most cost-effective method tested (order-6 plain Filon wins
on all three). This contrasts sharply with CNOT3, where a clean rotating frame
lets the cheap-per-step Hermite method win — exactly the dividing line argued
earlier: Filon's value is on problems where the oscillation can't be transformed
away.

**Caveat**: the baseline here is RK4, a general-purpose method. The stronger
incumbent — QGD's high-order Hermite, which is itself oscillation-capable and very
cheap per step — was **not** run on these problems (its fixed-carrier control
model does not cleanly express a chirp or a Bessel comb, and wiring custom
high-order-differentiable controls into QGD is nontrivial). So the honest claim is
"Filon beats a standard RK4 baseline and is stable in the oscillatory regime"; a
Filon-vs-Hermite comparison on these problems is the natural next step.
