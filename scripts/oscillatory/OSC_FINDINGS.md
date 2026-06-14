# Is Filon suitable for problems with no clean rotating frame?

Convergence study of `filon_solve_hardcoded` / `controlled_filon_solve` (s=0,1,2,
orders 2/4/6) on three problems where the oscillation cannot be analytically
transformed away. The complement to CNOT3 (clean rotating frame → Hermite wins;
[[cnot3-frame-campaign-results]]).

Scripts: `osc_compute.jl` (3 problems → `data/oscillatory/osc_data.jls`),
`osc_omega.jl` (ω-scaling → `osc_omega.jls`), `osc_plot.jl`
(→ `plots/oscillatory/osc_convergence.png`, `osc_omega.png`), `osc_report.jl`.
Run on SLURM via `srun`, self-contained in the umbrella env.

**Methods.** Subjects: Filon, controlled-Filon (s=0,1,2). Competitors:
**Gauss–Legendre implicit RK at matched orders 2/4/6** (structure-preserving —
unitary for skew-Hermitian, A-stable — a strong fair baseline) and classical
**RK4**. Ground truth = fine independent RK4 (cross-checked vs fine Filon s=2 to
1e-11–1e-13). Errors = final-time 2-norm. Control derivatives analytic,
FD-validated (≤2e-6).

Problems (drift diagonal constant → Filon ansatz = bare levels): **P1** parametric
flux modulation (2 qubits, Bessel-comb coupling), **P2** chirped drive (2-level),
**P3** few-cycle Gaussian (2-level, envelope on the carrier timescale).

## Results

**Orders confirmed** — Filon, controlled-Filon, and GL all hit their design
orders 2/4/6 (floor ~1e-13).

**Fair, matched-order comparison** (this is the key point — *not* "Filon wins
because it is higher order"). Cheapest wall-clock to reach err ≤ 1e-10, Filon vs
the same-order Gauss–Legendre IRK:

| order | problem    | Filon      | GL (same order) | Filon speed-up |
|-------|------------|------------|-----------------|----------------|
| 6     | parametric | 1.36e-3 s  | 9.23e-3 s       | 6.8×           |
| 6     | chirp      | 1.19e-3 s  | 2.45e-3 s       | 2.1×           |
| 6     | few-cycle  | 4.39e-4 s  | 1.58e-3 s       | 3.6×           |
| 4     | parametric | 6.77e-3 s  | 2.73e-2 s       | 4.0×           |
| 4     | chirp      | 4.48e-3 s  | 2.40e-2 s       | 5.4×           |
| 4     | few-cycle  | 2.87e-3 s  | 1.22e-2 s       | 4.3×           |

**At every matched order Filon beats GL by ~2–7×.** Two causes: (a) Filon's
implicit step solves an N×N system, GL order-2m an (mN)×(mN) one; (b) the ansatz
removes the diagonal phase, so Filon needs fewer steps. Filon s=2 is the overall
fastest method on all three problems.

**RK4** (explicit, order 4) is cheap-per-step when stable but **blows up at coarse
steps** (chirp: err 2e11 at ω₀Δt≈6); every Filon/GL variant stays bounded.

**Controlled vs plain Filon**: cFilon improves *per-step* coarse accuracy
(chirp s=1: 9×) but is **less cost-effective** than plain Filon at matched order
(extra carrier terms cost more than they buy). Plain Filon s=2 is the pick here.

**ω-scaling** (`osc_omega.jl`: fast diagonal ω₀ + slow transverse coupling, sweep
ω₀): **all methods scale ∝ ω₀** — including Filon. Filon does *not* achieve
ω₀-independent step counts: the off-diagonal coupling drives each component's
*envelope* at the level-spacing frequency (=ω₀), and Filon interpolates that
envelope polynomially, so it must resolve ω₀. Filon simply has the **smallest
constant** (~2× fewer steps than GL6, ~8× fewer than RK4; ~5–10× cheaper
wall-clock than GL6 from its smaller per-step solve). True ω₀-*independence* would
require the coupling to carry a carrier matching the level spacing so cFilon's
modified ansatz absorbs it — i.e. the CNOT3 rotating-frame structure.

## Verdict

**Yes — Filon is suitable and effective on these no-clean-rotating-frame
problems**, but the honest characterisation is a **robust constant-factor
efficiency gain (~3–10×) over matched-order structure-preserving methods**, plus
stability where explicit methods blow up. It is *not* an improved asymptotic
scaling with the oscillation frequency (unless the couplings are carrier-resonant,
which is the regime already covered by CNOT3). This still favours Filon: order-6
plain Filon is the fastest method tested on every problem.

## Where else Filon may be effective (candidates)

- **NMR / ESR / Larmor precession**: strong static field (fast diagonal) + slow or
  RF transverse drive — exactly the constant-factor regime demonstrated here.
- **Multi-tone / Floquet / frequency-comb driving**: several *known* carriers →
  controlled-Filon's multi-carrier ansatz is its designed strength (untested here).
- **Near-resonant couplings / interaction picture with small detuning**: if a
  coupling's carrier matches a level spacing, cFilon can absorb it → a genuine
  *scaling* win (the case to test next).
- **Trapped-ion sidebands, cavity QED with large detunings**, and large linear
  oscillatory systems (semidiscretised wave/Schrödinger) with a fast known
  spectral diagonal.
- **Not** where a clean rotating frame removes all oscillation (CNOT3 rwa): there
  the cheap-per-step Hermite method wins.

## Caveat (baseline)

Competitors are RK4 and Gauss–Legendre IRK. QGD's high-order Hermite was not run
(its fixed-carrier control model does not cleanly express a chirp or Bessel comb).
A Filon-vs-Hermite comparison on these problems is the natural next step.
